import coremltools as ct
import numpy as np
import torch
from executorch.examples.models.llama.export_llama_lib import (
    _prepare_for_llama_export,
    build_args_parser,
)
from numpy import dtype

parser = build_args_parser()
args = parser.parse_args()

model_manager = _prepare_for_llama_export("llama2", args)

model = model_manager.model
model.eval()


def get_example_inputs(max_batch_size, args, coreml=False, use_enumerated_shapes=False):
    tokens = torch.tensor([[1 for _ in range(max_batch_size)]], dtype=torch.long)
    if use_enumerated_shapes:
        ct_tokens_shape = ct.EnumeratedShapes(
            shapes=[
                [1, 1],
                [1, max_batch_size],
            ],
            default=[1, max_batch_size],
        )
    else:
        ct_tokens_shape = ct.Shape([1, max_batch_size])

    ct_tokens = ct.TensorType(
        shape=ct_tokens_shape,
        dtype=np.int64,
    )

    if args.use_kv_cache:
        # NOTE: torch.jit.trace does not work if tensor has size 1, but ct.convert does not work if not 512, so for KV cache with batch input, size should be 1
        # input_pos = torch.tensor([0 for _ in range(max_batch_size)], dtype=torch.long)
        input_pos = torch.tensor([0], dtype=torch.long)
        ct_input_pos = ct.TensorType(shape=ct.Shape([1]), dtype=np.int64)

        if coreml:
            return (ct_tokens, ct_input_pos)
        return (tokens, input_pos)

    if coreml:
        return (ct_tokens,)
    return (tokens,)


# Batch with kv cache runs into issues
# Either we need input_pos to be size batch_size to export with jit.trace or we need it to be size 1 to export with ct.convert
# Might try refactoring the model so that jit.trace works when it is size 1 (interested as starting position)
if args.use_kv_cache:
    max_batch_size = 1
else:
    max_batch_size = 128

example_inputs = get_example_inputs(max_batch_size, args)

print("Example input shapes: ", [t.shape for t in example_inputs])

traced_model = torch.jit.trace(model, example_inputs)

states = None
if args.use_kv_cache:
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=v[1].shape,
            ),
            name=v[0],
        )
        for v in traced_model.named_buffers()
        if v[0].endswith("_cache")
    ]

mlmodel = ct.convert(
    traced_model,
    inputs=list(
        get_example_inputs(
            max_batch_size=max_batch_size,
            args=args,
            coreml=True,
            use_enumerated_shapes=True,
        )
    ),
    outputs=[ct.TensorType(name="op")],
    states=states,
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)

mlmodel.save(args.output_name)
