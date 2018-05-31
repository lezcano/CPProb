# automatically generated by the FlatBuffers compiler, do not modify

# namespace: protocol

import flatbuffers

class Beta(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsBeta(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Beta()
        x.Init(buf, n + offset)
        return x

    # Beta
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Beta
    def Mode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    # Beta
    def Certainty(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

    # Beta
    def Value(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float64Flags, o + self._tab.Pos)
        return 0.0

def BetaStart(builder): builder.StartObject(3)
def BetaAddMode(builder, mode): builder.PrependFloat64Slot(0, mode, 0.0)
def BetaAddCertainty(builder, certainty): builder.PrependFloat64Slot(1, certainty, 0.0)
def BetaAddValue(builder, value): builder.PrependFloat64Slot(2, value, 0.0)
def BetaEnd(builder): return builder.EndObject()