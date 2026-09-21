"""Resize static fully-convolutional exports while freezing quantization.

Used only for Vela shape screening. The selected deployment model is separately
calibrated with all 64 images. Refuse unrecognized spatial reshape patterns.
"""
from pathlib import Path
import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


def resize(source,dest,height,width):
    m=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(Path(source).read_bytes(),0))
    assert len(m.subgraphs)==1
    g=m.subgraphs[0];assert len(g.inputs)==1
    old_h,old_w=map(int,g.tensors[g.inputs[0]].shape[1:3])
    def spatial(value,old,new):
        if value==1:return 1
        # All accepted models use powers-of-two spatial downsampling.
        divisors=[d for d in (1,2,4,8,16,32) if (old+d-1)//d==value]
        assert len(divisors)==1,(value,old,divisors)
        d=divisors[0];return (new+d-1)//d
    for t in g.tensors:
        data=m.buffers[t.buffer].data
        if data is not None and len(data):continue
        if len(t.shape)==4:
            # Global pooling and ECA's channel-axis 1D convolution are not
            # spatial maps. Accepted scan sizes keep all spatial heights > 1.
            if t.shape[1]==1:continue
            t.shape=np.array([t.shape[0],spatial(int(t.shape[1]),old_h,height),spatial(int(t.shape[2]),old_w,width),t.shape[3]],np.int32)
        elif len(t.shape)>4:raise ValueError(('spatial reshape requires explicit handling',t.name,t.shape))
    updates={}
    for op in g.operators:
        code=m.operatorCodes[op.opcodeIndex].builtinCode
        output=g.tensors[op.outputs[0]]
        if code==schema.BuiltinOperator.TRANSPOSE_CONV:
            idx=op.inputs[0];value=output.shape
        elif code in (schema.BuiltinOperator.RESIZE_BILINEAR,schema.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR):
            idx=op.inputs[1];value=output.shape[1:3]
        elif code==schema.BuiltinOperator.RESHAPE and len(op.inputs)>1:
            idx=op.inputs[1];value=output.shape
        else:continue
        tensor=g.tensors[idx];assert tensor.type==schema.TensorType.INT32
        raw=np.asarray(value,dtype='<i4').tobytes()
        if tensor.buffer in updates:assert updates[tensor.buffer]==raw
        updates[tensor.buffer]=raw
    for idx,raw in updates.items():m.buffers[idx].data=np.frombuffer(raw,dtype=np.uint8)
    builder=flatbuffers.Builder(0);builder.Finish(m.Pack(builder),file_identifier=b'TFL3')
    Path(dest).write_bytes(builder.Output())
