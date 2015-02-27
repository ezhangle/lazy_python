from abc import ABCMeta, abstractproperty
from dis import Bytecode, opname, opmap, hasjabs, hasjrel
from operator import is_, not_
from types import CodeType, FunctionType

from lazy._thunk import strict, thunk


def _sparse_args(instrs):
    """
    Makes the arguments sparse so that instructions live at the correct
    index for the jump resolution step.
    The `None` instructions will be filtered out.
    """
    for instr in instrs:
        yield instr
        if instr.arg is not None:
            yield None
            yield None


class ops(dict):
    def __getattr__(self, name):
        return self[name]

ops = ops(opmap)


class CodeTransformer(object, metaclass=ABCMeta):
    """
    A code object transformer, simmilar to the AstTransformer from the ast
    module.
    """
    @abstractproperty
    def stack_modifier(self):
        """
        How much does this transformer affect the maximum stack usage.
        """
        return 0

    def __init__(self):
        self._instrs = None
        self._consts = None

    def __getitem__(self, idx):
        return self._instrs[idx]

    def index(self, instr):
        """
        Returns the index of an `Instruction`.
        """
        return self._instrs.index(instr)

    def __iter__(self):
        return iter(self._instrs)

    def const_index(self, obj):
        """
        The index of a constant.
        If `obj` is not already a constant, it will be added to the consts
        and given a new const index.
        """
        try:
            return self._consts[obj]
        except KeyError:
            self._consts[obj] = ret = self._const_idx
            self._const_idx += 1
            return ret

    def visit_generic(self, instr):
        if instr is None:
            yield None
            return

        yield from getattr(self, 'visit_' + instr.opname, lambda *a: a)(instr)

    def visit_const(self, const):
        """
        Override this method to transform the `co_consts` of the code object.
        """
        if isinstance(const, CodeType):
            return type(self).visit(const)
        else:
            return const

    def _id(self, obj):
        """
        Identity function.
        """
        return obj

    visit_name = _id
    visit_varname = _id
    visit_freevar = _id
    visit_cellvar = _id
    visit_default = _id

    del _id

    def visit(self, co, name=None):
        """
        Visit a code object, applying the transforms.
        """
        # WARNING:
        # This is setup in this double assignment way because jump args
        # must backreference their original jump target before any transforms.
        # Don't refactor this into a single pass.
        self._instrs = list(_sparse_args([
            Instruction(b.opcode, b.arg) for b in Bytecode(co)
        ]))
        self._instrs = [
            instr and instr._with_jmp_arg(self) for instr in self._instrs
        ]
        self._consts = {
            self.visit_const(k): idx for idx, k in enumerate(co.co_consts)
        }
        self._const_idx = len(co.co_consts)

        self._instrs = sum(
            (tuple(self.visit_generic(_instr)) for _instr in self),
            (),
        )

        return CodeType(
            co.co_argcount,
            co.co_kwonlyargcount,
            co.co_nlocals,
            co.co_stacksize + self.stack_modifier,
            co.co_flags,
            b''.join(
                (instr or b'') and instr.to_bytecode(self) for instr in self
            ),
            tuple(sorted(self._consts, key=lambda c: self._consts[c])),
            tuple(self.visit_name(n) for n in co.co_names),
            tuple(self.visit_varname(n) for n in co.co_varnames),
            co.co_filename,
            name if name is not None else co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            tuple(self.visit_freevar(c) for c in co.co_freevars),
            tuple(self.visit_cellvar(c) for c in co.co_cellvars),
        )

    def __repr__(self):
        return '<{cls}: {instrs!r}>'.format(
            cls=type(self).__name__,
            instrs=self._instrs,
        )


class Instruction(object):
    """
    An abstraction of an instruction.
    This must live in a code object.
    """
    def __init__(self, opcode, arg=None):
        self.opcode = opcode
        self.arg = arg
        self.reljmp = False
        self.absjmp = False
        self._stolen = None

    def _with_jmp_arg(self, transformer):
        """
        If this is a jump opcode, then convert the arg to the instruction
        to jump to.
        """
        opcode = self.opcode
        if opcode in hasjrel:
            self.arg = transformer[self.index(transformer) + 2 + self.arg]
            self.reljmp = True
        elif opcode in hasjabs:
            self.arg = transformer[self.arg]
            self.absjmp = True
        return self

    @property
    def opname(self):
        return opname[self.opcode]

    def to_bytecode(self, transformer):
        bs = bytes((self.opcode,))
        arg = self.arg
        if isinstance(arg, Instruction):
            if self.absjmp:
                bs += arg.jmp_index(transformer).to_bytes(2, 'little')
            elif self.reljmp:
                bs += (
                    arg.jmp_index(transformer) - self.index(transformer)
                ).to_bytes(2, 'little')
            else:
                raise ValueError('must be relative or absolute jump')
        elif arg is not None:
            bs += arg.to_bytes(2, 'little')
        return bs

    def index(self, transformer):
        return transformer.index(self)

    def jmp_index(self, transformer):
        return transformer.index(self._stolen or self)

    def __repr__(self):
        arg = self.arg
        return '<{cls}: {opname}{arg}>'.format(
            cls=type(self).__name__,
            opname=self.opname,
            arg=': ' + str(arg) if self.arg is not None else '',
        )

    def steal(self, instr):
        instr._stolen = self
        return self


def _lazy_is(a, b, *, is_=is_):
    return thunk(is_, a, b)


def _lazy_not(a, *, not_=not_):
    return thunk(not_, a)


class LazyTransformer(CodeTransformer):
    @property
    def stack_modifier(self):
        return 1

    def visit_const(self, const):
        const = super().visit_const(const)
        if not isinstance(const, CodeType):
            const = thunk.fromvalue(const)
        return const

    visit_freevar = thunk.fromvalue
    visit_cellvar = thunk.fromvalue

    def visit_MAKE_FUNCTION(self, instr):
        """
        Functions should have strict names.
        """
        yield Instruction(
            ops.LOAD_CONST, self.const_index(strict)
        ).steal(instr)
        # TOS = strict
        # TOS1 = func_name

        yield Instruction(ops.ROT_TWO)
        # TOS = func_name
        # TOS1 = strict

        yield Instruction(ops.CALL_FUNCTION, 1)
        # TOS = strict(func_name)

        yield instr
        # TOS = new_function

    visit_MAKE_CLOSURE = visit_MAKE_FUNCTION

    def _visit_load_name(self, instr):
        """
        Loading a name immediatly wraps it in a `thunk`.
        """
        yield Instruction(
            ops.LOAD_CONST, self.const_index(thunk.fromvalue)
        ).steal(instr)
        # TOS = thunk.fromvalue

        yield instr
        # TOS = value
        # TOS1 = thunk.fromvalue

        yield Instruction(ops.CALL_FUNCTION, 1)
        # TOS = thunk.fromvalue(value)

    visit_LOAD_NAME = visit_LOAD_GLOBAL = visit_LOAD_FAST = _visit_load_name

    def visit_COMPARE_OP(self, instr):
        """
        Replace the `is` operator to act on the values the thunks represent.
        This makes `is` lazy.
        """
        if instr.arg != 8: # is
            yield from self.visit_generic(instr)
            return

        yield Instruction(
            ops.LOAD_CONST, self.const_index(_lazy_is)
        ).steal(instr)
        # TOS = _lazy_is
        # TOS1 = a
        # TOS2 = b

        # This safe to do because `is` is commutative 100% of the time.
        # We are doing a pointer compare so we can move the operands around.
        # This saves us from doing an extra ROT_TWO to preserve the order.
        yield Instruction(ops.ROT_THREE)
        # TOS = a
        # TOS1 = b
        # TOS2 = _lazy_is

        yield Instruction(ops.CALL_FUNCTION, 2)
        # TOS = _lazy_is(b, a)

    def visit_UNARY_NOT(self, instr):
        """
        Replace the `not` operator to act on the values that the thunks
        represent.
        This makes `not` lazy.
        """
        yield Instruction(
            ops.LOAD_CONST, self.const_index(_lazy_not)
        ).steal(instr)
        # TOS = _lazy_not
        # TOS1 = arg

        yield Instruction(ops.ROT_TWO)
        # TOS = arg
        # TOS1 = _lazy_not

        yield Instruction(ops.CALL_FUNCTION, 1)
        # TOS = _lazy_not(arg)


def lazy_function(f):
    return thunk.fromvalue(
        FunctionType(
            LazyTransformer().visit(f.__code__),
            f.__globals__,
            f.__name__,
            tuple(map(thunk.fromvalue, f.__defaults__ or ())),
            f.__closure__,
        ),
    )
