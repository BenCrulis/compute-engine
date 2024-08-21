
#include "tensorflow/core/framework/op.h" // for registration of custom op
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "larq_compute_engine/core/ternary/ternary.h"


using namespace std;
using namespace tensorflow;
using namespace shape_inference;
using shape_inference::ShapeHandle;


REGISTER_OP("TernaryMatmul")
    .Input("input: float32")
    .Input("weights: uint8")
    .Input("scale: float32")
    .Input("bias: float32")
    .Input("clamp: float32")
    .Output("output: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

      // get the shape of the first input tensor (input index 0)
      ShapeHandle input_shape_0;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &input_shape_0));

      // Get the shape of the second input tensor (input index 1)
      ShapeHandle input_shape_1;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_shape_1));

      ShapeHandle input_shape_2;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input_shape_2));

      ShapeHandle input_shape_3;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &input_shape_3));

      ShapeHandle input_shape_4;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &input_shape_4));

      // Get the last dimension of the first input tensor
      DimensionHandle last_dim = c->Dim(input_shape_0, -1);

      DimensionHandle weight_shape_input = c->Dim(input_shape_1, 1);
      DimensionHandle weight_shape_output = c->Dim(input_shape_1, 0);

      int out_dim;
      if (c->ValueKnown(weight_shape_output)) {
        out_dim = c->Value(weight_shape_output);
      } else {
        return errors::InvalidArgument("Last dimension size of input2 must be known.");
      }

      int in_dim;
      if (c->ValueKnown(last_dim)) {
        in_dim = c->Value(last_dim);
      } else {
        return errors::InvalidArgument("Last dimension size of input1 must be known.");
      }


      DimensionHandle scale_size = c->Dim(input_shape_2, 0);
      DimensionHandle bias_size = c->Dim(input_shape_3, 0);
      
      if (c->ValueKnown(scale_size)) {
        int64 scale_size_val = c->Value(scale_size);

        if (scale_size_val != out_dim) {
          return errors::InvalidArgument("Scale dimension must be equal to the number of outputs.");
        }
      } else {
        return errors::InvalidArgument("Scale dimension must be known.");
      }

      if (c->ValueKnown(bias_size)) {
        int64 bias_size_val = c->Value(bias_size);
        if (bias_size_val != out_dim) {
          return errors::InvalidArgument("Bias dimension must be equal to the number of outputs.");
        }
      } else {
        return errors::InvalidArgument("Bias dimension must be known.");
      }

      if (c->RankKnown(input_shape_4)) {
        if (c->Rank(input_shape_4) != 0) {
          return errors::InvalidArgument("Clamp must be a scalar.");
        }
      } else {
        return errors::InvalidArgument("Clamp Rank must be known.");
      }

      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape_0, -1, weight_shape_output, &output_shape));

      // Define the output shape (example: similar to the first input shape)
      // ShapeHandle output_shape = c->MakeShape({c->Dim(input_shape_0, 0), last_dim});
      
      // Set the output shape
      c->set_output(0, output_shape);
      //return Status::OK();
      return OkStatus();
    });



class TernaryMatmulOp : public OpKernel {
 public:
  explicit TernaryMatmulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get the input tensors
    const Tensor& input1 = context->input(0);
    const Tensor& input2 = context->input(1);
    // const Tensor& scale = context->input(2);
    // const Tensor& bias = context->input(3);
    // const Tensor& clamp = context->input(4);

    // Get the shapes of the input tensors
    const TensorShape& shape1 = input1.shape();
    const TensorShape& shape2 = input2.shape();

    // Get the first dimension of the second input tensor
    int64 last_dim_size = shape2.dim_size(0);

    // Define the shape of the output tensor by copying the input shape and changing the last dimension
    TensorShape output_shape = shape1;
    output_shape.set_dim(output_shape.dims() - 1, last_dim_size);

    // Preallocate the output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    // Perform computation (this example just sets output to zero)
    auto output_flat = output_tensor->flat<float>();
    output_flat.setZero();
  }
};



///////////////////////////////////////////////////////////////////////////////
// ternary unpacking

REGISTER_OP("UnpackTernary")
    .Input("packed: uint8")
    .Input("target_size: int32")
    .Output("output: float32")
    .SetDoNotOptimize()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

      // get the shape of the first input tensor (input index 0)
      ShapeHandle input_shape_0;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape_0));

      ShapeHandle input_shape_1;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &input_shape_1));

      DimensionHandle weight_shape_input = c->Dim(input_shape_0, 1);
      DimensionHandle weight_shape_output = c->Dim(input_shape_0, 0);

      int out_dim;
      if (c->ValueKnown(weight_shape_output)) {
        out_dim = c->Value(weight_shape_output);
      } else {
        return errors::InvalidArgument("Last dimension size of input1 must be known.");
      }

      int in_dim;
      if (c->ValueKnown(weight_shape_input)) {
        in_dim = c->Value(weight_shape_input);
      } else {
        return errors::InvalidArgument("Last dimension size of input1 must be known.");
      }

      DimensionHandle target_dim = c->Dim(input_shape_1, 0);

      int unpacked_dim = c->Value(target_dim);

      int raw_unpacked_size = in_dim * 4;

      int remainder = unpacked_dim % 4;

      // if (!(((unpacked_dim / 4 == in_dim) && (remainder == 0))
      //     || (((unpacked_dim / 4 + 1) == in_dim) && (remainder != 0)))) {
      //   return errors::InvalidArgument("target unpacked size is not compatible with packed input size");
      // }

      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape_0, -1, c->MakeDim(-1), &output_shape));

      // Define the output shape (example: similar to the first input shape)
      // ShapeHandle output_shape = c->MakeShape({c->Dim(input_shape_0, 0), last_dim});
      
      // Set the output shape
      c->set_output(0, output_shape);
      //return Status::OK();
      return OkStatus();
    });



class UnpackTernaryOp : public OpKernel {
 public:
  explicit UnpackTernaryOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Get the input tensors
    const Tensor& input1 = context->input(0);
    const Tensor& input2 = context->input(1);

    // Get the shapes of the input tensor
    const TensorShape& shape1 = input1.shape();
    const TensorShape& shape2 = input2.shape();

    int in_dim = shape1.dim_size(1);

    // int unpacked_dim = 4*shape1.dim_size(1);
    int unpacked_dim = input2.flat<int>()(0);

    int remainder = unpacked_dim % 4;

    OP_REQUIRES(context, ((unpacked_dim / 4 == in_dim) && (remainder == 0))
                      || (((unpacked_dim / 4 + 1) == in_dim) && (remainder != 0)),
                errors::InvalidArgument("incorrect target size"));

    // Define the shape of the output tensor by copying the input shape and changing the last dimension
    TensorShape output_shape = shape1;
    output_shape.set_dim(1, unpacked_dim);

    // Preallocate the output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    uint8* input_data = reinterpret_cast<uint8*>(input1.data());
    float* output_data = reinterpret_cast<float*>(output_tensor->data());

    // Perform computation (this example just sets output to zero)
    auto output_flat = output_tensor->flat<float>();
    output_flat.setZero();

    unpack_ternary(input_data, output_data, unpacked_dim, shape1.dim_size(0));
  }
};


REGISTER_KERNEL_BUILDER(Name("TernaryMatmul").Device(DEVICE_CPU), TernaryMatmulOp);
REGISTER_KERNEL_BUILDER(Name("UnpackTernary").Device(DEVICE_CPU), UnpackTernaryOp);
