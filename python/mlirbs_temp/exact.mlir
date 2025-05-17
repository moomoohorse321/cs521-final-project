module {
  func.func @basic_matmul(%arg0: tensor<4x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<1x3xf32> {
    %0 = "tf.Const"() {value = dense<"lhs[0, 2] * rhs[2, 0] = Tensor(\22strided_slice_10:0\22, shape=(), dtype=float32) * Tensor(\22strided_slice_11:0\22, shape=(), dtype=float32)"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %1 = "tf.Const"() {value = dense<"lhs[0, 1] * rhs[1, 0] = Tensor(\22strided_slice_6:0\22, shape=(), dtype=float32) * Tensor(\22strided_slice_7:0\22, shape=(), dtype=float32)"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %2 = "tf.Const"() {value = dense<"lhs[0, 0] * rhs[0, 0] = Tensor(\22strided_slice_2:0\22, shape=(), dtype=float32) * Tensor(\22strided_slice_3:0\22, shape=(), dtype=float32)"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    "tf.PrintV2"(%2) {device = "", end = "\0A", output_stream = "stderr"} : (tensor<!tf_type.string>) -> ()
    "tf.PrintV2"(%1) {device = "", end = "\0A", output_stream = "stderr"} : (tensor<!tf_type.string>) -> ()
    "tf.PrintV2"(%0) {device = "", end = "\0A", output_stream = "stderr"} : (tensor<!tf_type.string>) -> ()
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<1x3xf32>
    %c = stablehlo.constant dense<0> : tensor<1x2xi32>
    %c_0 = stablehlo.constant dense<[[1, 0]]> : tensor<1x2xi32>
    %c_1 = stablehlo.constant dense<1> : tensor<1x2xi32>
    %c_2 = stablehlo.constant dense<[[1, 2]]> : tensor<1x2xi32>
    %c_3 = stablehlo.constant dense<[[2, 0]]> : tensor<1x2xi32>
    %c_4 = stablehlo.constant dense<[[2, 1]]> : tensor<1x2xi32>
    %c_5 = stablehlo.constant dense<2> : tensor<1x2xi32>
    %c_6 = stablehlo.constant dense<[[3, 0]]> : tensor<1x2xi32>
    %c_7 = stablehlo.constant dense<[[0, 1]]> : tensor<1x2xi32>
    %c_8 = stablehlo.constant dense<[[3, 1]]> : tensor<1x2xi32>
    %c_9 = stablehlo.constant dense<[[3, 2]]> : tensor<1x2xi32>
    %c_10 = stablehlo.constant dense<[[0, 2]]> : tensor<1x2xi32>
    %3 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %4 = stablehlo.reshape %3 : (tensor<1x1xf32>) -> tensor<f32>
    %5 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %6 = stablehlo.reshape %5 : (tensor<1x1xf32>) -> tensor<f32>
    %7 = stablehlo.multiply %4, %6 : tensor<f32>
    %8 = stablehlo.reshape %7 : (tensor<f32>) -> tensor<1xf32>
    %9 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %10 = stablehlo.reshape %9 : (tensor<1x1xf32>) -> tensor<f32>
    %11 = stablehlo.slice %arg1 [0:1, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %12 = stablehlo.reshape %11 : (tensor<1x1xf32>) -> tensor<f32>
    %13 = stablehlo.multiply %10, %12 : tensor<f32>
    %14 = stablehlo.reshape %13 : (tensor<f32>) -> tensor<1xf32>
    %15 = stablehlo.slice %arg0 [0:1, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %16 = stablehlo.reshape %15 : (tensor<1x1xf32>) -> tensor<f32>
    %17 = stablehlo.slice %arg1 [1:2, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x1xf32>) -> tensor<f32>
    %19 = stablehlo.multiply %16, %18 : tensor<f32>
    %20 = stablehlo.reshape %19 : (tensor<f32>) -> tensor<1xf32>
    %21 = stablehlo.slice %arg0 [0:1, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %22 = stablehlo.reshape %21 : (tensor<1x1xf32>) -> tensor<f32>
    %23 = stablehlo.slice %arg1 [2:3, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %24 = stablehlo.reshape %23 : (tensor<1x1xf32>) -> tensor<f32>
    %25 = stablehlo.multiply %22, %24 : tensor<f32>
    %26 = stablehlo.reshape %25 : (tensor<f32>) -> tensor<1xf32>
    %27 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %28 = stablehlo.reshape %27 : (tensor<1x1xf32>) -> tensor<f32>
    %29 = stablehlo.slice %arg1 [0:1, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %30 = stablehlo.reshape %29 : (tensor<1x1xf32>) -> tensor<f32>
    %31 = stablehlo.multiply %28, %30 : tensor<f32>
    %32 = stablehlo.reshape %31 : (tensor<f32>) -> tensor<1xf32>
    %33 = stablehlo.slice %arg0 [0:1, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %34 = stablehlo.reshape %33 : (tensor<1x1xf32>) -> tensor<f32>
    %35 = stablehlo.slice %arg1 [1:2, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %36 = stablehlo.reshape %35 : (tensor<1x1xf32>) -> tensor<f32>
    %37 = stablehlo.multiply %34, %36 : tensor<f32>
    %38 = stablehlo.reshape %37 : (tensor<f32>) -> tensor<1xf32>
    %39 = stablehlo.slice %arg0 [0:1, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x1xf32>) -> tensor<f32>
    %41 = stablehlo.slice %arg1 [2:3, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %42 = stablehlo.reshape %41 : (tensor<1x1xf32>) -> tensor<f32>
    %43 = stablehlo.multiply %40, %42 : tensor<f32>
    %44 = stablehlo.reshape %43 : (tensor<f32>) -> tensor<1xf32>
    %45 = stablehlo.slice %arg0 [1:2, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %46 = stablehlo.reshape %45 : (tensor<1x1xf32>) -> tensor<f32>
    %47 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %48 = stablehlo.reshape %47 : (tensor<1x1xf32>) -> tensor<f32>
    %49 = stablehlo.multiply %46, %48 : tensor<f32>
    %50 = stablehlo.reshape %49 : (tensor<f32>) -> tensor<1xf32>
    %51 = stablehlo.slice %arg0 [1:2, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %52 = stablehlo.reshape %51 : (tensor<1x1xf32>) -> tensor<f32>
    %53 = stablehlo.slice %arg1 [1:2, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %54 = stablehlo.reshape %53 : (tensor<1x1xf32>) -> tensor<f32>
    %55 = stablehlo.multiply %52, %54 : tensor<f32>
    %56 = stablehlo.reshape %55 : (tensor<f32>) -> tensor<1xf32>
    %57 = stablehlo.slice %arg0 [1:2, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %58 = stablehlo.reshape %57 : (tensor<1x1xf32>) -> tensor<f32>
    %59 = stablehlo.slice %arg1 [2:3, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %60 = stablehlo.reshape %59 : (tensor<1x1xf32>) -> tensor<f32>
    %61 = stablehlo.multiply %58, %60 : tensor<f32>
    %62 = stablehlo.reshape %61 : (tensor<f32>) -> tensor<1xf32>
    %63 = stablehlo.slice %arg0 [1:2, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %64 = stablehlo.reshape %63 : (tensor<1x1xf32>) -> tensor<f32>
    %65 = stablehlo.slice %arg1 [0:1, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %66 = stablehlo.reshape %65 : (tensor<1x1xf32>) -> tensor<f32>
    %67 = stablehlo.multiply %64, %66 : tensor<f32>
    %68 = stablehlo.reshape %67 : (tensor<f32>) -> tensor<1xf32>
    %69 = stablehlo.slice %arg0 [1:2, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %70 = stablehlo.reshape %69 : (tensor<1x1xf32>) -> tensor<f32>
    %71 = stablehlo.slice %arg1 [1:2, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %72 = stablehlo.reshape %71 : (tensor<1x1xf32>) -> tensor<f32>
    %73 = stablehlo.multiply %70, %72 : tensor<f32>
    %74 = stablehlo.reshape %73 : (tensor<f32>) -> tensor<1xf32>
    %75 = stablehlo.slice %arg0 [1:2, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %76 = stablehlo.reshape %75 : (tensor<1x1xf32>) -> tensor<f32>
    %77 = stablehlo.slice %arg1 [2:3, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %78 = stablehlo.reshape %77 : (tensor<1x1xf32>) -> tensor<f32>
    %79 = stablehlo.multiply %76, %78 : tensor<f32>
    %80 = stablehlo.reshape %79 : (tensor<f32>) -> tensor<1xf32>
    %81 = stablehlo.slice %arg0 [1:2, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %82 = stablehlo.reshape %81 : (tensor<1x1xf32>) -> tensor<f32>
    %83 = stablehlo.slice %arg1 [0:1, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %84 = stablehlo.reshape %83 : (tensor<1x1xf32>) -> tensor<f32>
    %85 = stablehlo.multiply %82, %84 : tensor<f32>
    %86 = stablehlo.reshape %85 : (tensor<f32>) -> tensor<1xf32>
    %87 = stablehlo.slice %arg0 [1:2, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %88 = stablehlo.reshape %87 : (tensor<1x1xf32>) -> tensor<f32>
    %89 = stablehlo.slice %arg1 [1:2, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %90 = stablehlo.reshape %89 : (tensor<1x1xf32>) -> tensor<f32>
    %91 = stablehlo.multiply %88, %90 : tensor<f32>
    %92 = stablehlo.reshape %91 : (tensor<f32>) -> tensor<1xf32>
    %93 = stablehlo.slice %arg0 [0:1, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %94 = stablehlo.reshape %93 : (tensor<1x1xf32>) -> tensor<f32>
    %95 = stablehlo.slice %arg0 [1:2, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %96 = stablehlo.reshape %95 : (tensor<1x1xf32>) -> tensor<f32>
    %97 = stablehlo.slice %arg1 [2:3, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %98 = stablehlo.reshape %97 : (tensor<1x1xf32>) -> tensor<f32>
    %99 = stablehlo.multiply %96, %98 : tensor<f32>
    %100 = stablehlo.reshape %99 : (tensor<f32>) -> tensor<1xf32>
    %101 = stablehlo.slice %arg0 [2:3, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %102 = stablehlo.reshape %101 : (tensor<1x1xf32>) -> tensor<f32>
    %103 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %104 = stablehlo.reshape %103 : (tensor<1x1xf32>) -> tensor<f32>
    %105 = stablehlo.multiply %102, %104 : tensor<f32>
    %106 = stablehlo.reshape %105 : (tensor<f32>) -> tensor<1xf32>
    %107 = stablehlo.slice %arg0 [2:3, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %108 = stablehlo.reshape %107 : (tensor<1x1xf32>) -> tensor<f32>
    %109 = stablehlo.slice %arg1 [1:2, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %110 = stablehlo.reshape %109 : (tensor<1x1xf32>) -> tensor<f32>
    %111 = stablehlo.multiply %108, %110 : tensor<f32>
    %112 = stablehlo.reshape %111 : (tensor<f32>) -> tensor<1xf32>
    %113 = stablehlo.slice %arg0 [2:3, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %114 = stablehlo.reshape %113 : (tensor<1x1xf32>) -> tensor<f32>
    %115 = stablehlo.slice %arg1 [2:3, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %116 = stablehlo.reshape %115 : (tensor<1x1xf32>) -> tensor<f32>
    %117 = stablehlo.multiply %114, %116 : tensor<f32>
    %118 = stablehlo.reshape %117 : (tensor<f32>) -> tensor<1xf32>
    %119 = stablehlo.slice %arg0 [2:3, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %120 = stablehlo.reshape %119 : (tensor<1x1xf32>) -> tensor<f32>
    %121 = stablehlo.slice %arg1 [0:1, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %122 = stablehlo.reshape %121 : (tensor<1x1xf32>) -> tensor<f32>
    %123 = stablehlo.multiply %120, %122 : tensor<f32>
    %124 = stablehlo.reshape %123 : (tensor<f32>) -> tensor<1xf32>
    %125 = stablehlo.slice %arg1 [1:2, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %126 = stablehlo.reshape %125 : (tensor<1x1xf32>) -> tensor<f32>
    %127 = stablehlo.multiply %94, %126 : tensor<f32>
    %128 = stablehlo.reshape %127 : (tensor<f32>) -> tensor<1xf32>
    %129 = stablehlo.slice %arg0 [2:3, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %130 = stablehlo.reshape %129 : (tensor<1x1xf32>) -> tensor<f32>
    %131 = stablehlo.slice %arg1 [1:2, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %132 = stablehlo.reshape %131 : (tensor<1x1xf32>) -> tensor<f32>
    %133 = stablehlo.multiply %130, %132 : tensor<f32>
    %134 = stablehlo.reshape %133 : (tensor<f32>) -> tensor<1xf32>
    %135 = stablehlo.slice %arg0 [2:3, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %136 = stablehlo.reshape %135 : (tensor<1x1xf32>) -> tensor<f32>
    %137 = stablehlo.slice %arg1 [2:3, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %138 = stablehlo.reshape %137 : (tensor<1x1xf32>) -> tensor<f32>
    %139 = stablehlo.multiply %136, %138 : tensor<f32>
    %140 = stablehlo.reshape %139 : (tensor<f32>) -> tensor<1xf32>
    %141 = stablehlo.slice %arg0 [2:3, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %142 = stablehlo.reshape %141 : (tensor<1x1xf32>) -> tensor<f32>
    %143 = stablehlo.slice %arg1 [0:1, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %144 = stablehlo.reshape %143 : (tensor<1x1xf32>) -> tensor<f32>
    %145 = stablehlo.multiply %142, %144 : tensor<f32>
    %146 = stablehlo.reshape %145 : (tensor<f32>) -> tensor<1xf32>
    %147 = stablehlo.slice %arg0 [2:3, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %148 = stablehlo.reshape %147 : (tensor<1x1xf32>) -> tensor<f32>
    %149 = stablehlo.slice %arg1 [1:2, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %150 = stablehlo.reshape %149 : (tensor<1x1xf32>) -> tensor<f32>
    %151 = stablehlo.multiply %148, %150 : tensor<f32>
    %152 = stablehlo.reshape %151 : (tensor<f32>) -> tensor<1xf32>
    %153 = stablehlo.slice %arg0 [2:3, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %154 = stablehlo.reshape %153 : (tensor<1x1xf32>) -> tensor<f32>
    %155 = stablehlo.slice %arg1 [2:3, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %156 = stablehlo.reshape %155 : (tensor<1x1xf32>) -> tensor<f32>
    %157 = stablehlo.multiply %154, %156 : tensor<f32>
    %158 = stablehlo.reshape %157 : (tensor<f32>) -> tensor<1xf32>
    %159 = stablehlo.slice %arg0 [3:4, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %160 = stablehlo.reshape %159 : (tensor<1x1xf32>) -> tensor<f32>
    %161 = stablehlo.slice %arg1 [0:1, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %162 = stablehlo.reshape %161 : (tensor<1x1xf32>) -> tensor<f32>
    %163 = stablehlo.multiply %160, %162 : tensor<f32>
    %164 = stablehlo.reshape %163 : (tensor<f32>) -> tensor<1xf32>
    %165 = stablehlo.slice %arg0 [3:4, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %166 = stablehlo.reshape %165 : (tensor<1x1xf32>) -> tensor<f32>
    %167 = stablehlo.slice %arg1 [1:2, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %168 = stablehlo.reshape %167 : (tensor<1x1xf32>) -> tensor<f32>
    %169 = stablehlo.multiply %166, %168 : tensor<f32>
    %170 = stablehlo.reshape %169 : (tensor<f32>) -> tensor<1xf32>
    %171 = stablehlo.slice %arg0 [3:4, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %172 = stablehlo.reshape %171 : (tensor<1x1xf32>) -> tensor<f32>
    %173 = stablehlo.slice %arg1 [2:3, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %174 = stablehlo.reshape %173 : (tensor<1x1xf32>) -> tensor<f32>
    %175 = stablehlo.multiply %172, %174 : tensor<f32>
    %176 = stablehlo.reshape %175 : (tensor<f32>) -> tensor<1xf32>
    %177 = stablehlo.slice %arg0 [3:4, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %178 = stablehlo.reshape %177 : (tensor<1x1xf32>) -> tensor<f32>
    %179 = stablehlo.slice %arg1 [0:1, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %180 = stablehlo.reshape %179 : (tensor<1x1xf32>) -> tensor<f32>
    %181 = stablehlo.multiply %178, %180 : tensor<f32>
    %182 = stablehlo.reshape %181 : (tensor<f32>) -> tensor<1xf32>
    %183 = stablehlo.slice %arg0 [3:4, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %184 = stablehlo.reshape %183 : (tensor<1x1xf32>) -> tensor<f32>
    %185 = stablehlo.slice %arg1 [1:2, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %186 = stablehlo.reshape %185 : (tensor<1x1xf32>) -> tensor<f32>
    %187 = stablehlo.multiply %184, %186 : tensor<f32>
    %188 = stablehlo.reshape %187 : (tensor<f32>) -> tensor<1xf32>
    %189 = stablehlo.slice %arg0 [3:4, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %190 = stablehlo.reshape %189 : (tensor<1x1xf32>) -> tensor<f32>
    %191 = stablehlo.slice %arg1 [2:3, 1:2] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %192 = stablehlo.reshape %191 : (tensor<1x1xf32>) -> tensor<f32>
    %193 = stablehlo.multiply %190, %192 : tensor<f32>
    %194 = stablehlo.reshape %193 : (tensor<f32>) -> tensor<1xf32>
    %195 = stablehlo.slice %arg0 [3:4, 0:1] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %196 = stablehlo.reshape %195 : (tensor<1x1xf32>) -> tensor<f32>
    %197 = stablehlo.slice %arg1 [0:1, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %198 = stablehlo.reshape %197 : (tensor<1x1xf32>) -> tensor<f32>
    %199 = stablehlo.multiply %196, %198 : tensor<f32>
    %200 = stablehlo.reshape %199 : (tensor<f32>) -> tensor<1xf32>
    %201 = stablehlo.slice %arg0 [3:4, 1:2] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %202 = stablehlo.reshape %201 : (tensor<1x1xf32>) -> tensor<f32>
    %203 = stablehlo.slice %arg1 [1:2, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %204 = stablehlo.reshape %203 : (tensor<1x1xf32>) -> tensor<f32>
    %205 = stablehlo.multiply %202, %204 : tensor<f32>
    %206 = stablehlo.reshape %205 : (tensor<f32>) -> tensor<1xf32>
    %207 = stablehlo.slice %arg0 [3:4, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %208 = stablehlo.reshape %207 : (tensor<1x1xf32>) -> tensor<f32>
    %209 = stablehlo.slice %arg1 [2:3, 2:3] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %210 = stablehlo.reshape %209 : (tensor<1x1xf32>) -> tensor<f32>
    %211 = stablehlo.multiply %208, %210 : tensor<f32>
    %212 = stablehlo.reshape %211 : (tensor<f32>) -> tensor<1xf32>
    %213 = stablehlo.slice %arg0 [0:1, 2:3] : (tensor<4x3xf32>) -> tensor<1x1xf32>
    %214 = stablehlo.reshape %213 : (tensor<1x1xf32>) -> tensor<f32>
    %215 = stablehlo.slice %arg1 [2:3, 0:1] : (tensor<3x3xf32>) -> tensor<1x1xf32>
    %216 = stablehlo.reshape %215 : (tensor<1x1xf32>) -> tensor<f32>
    %217 = stablehlo.multiply %214, %216 : tensor<f32>
    %218 = stablehlo.reshape %217 : (tensor<f32>) -> tensor<1xf32>
    %219 = "stablehlo.scatter"(%cst, %c, %8) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %220 = "stablehlo.scatter"(%219, %c, %128) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %221 = "stablehlo.scatter"(%220, %c, %218) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %222 = "stablehlo.scatter"(%221, %c_7, %14) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %223 = "stablehlo.scatter"(%222, %c_7, %20) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %224 = "stablehlo.scatter"(%223, %c_7, %26) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %225 = "stablehlo.scatter"(%224, %c_10, %32) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %226 = "stablehlo.scatter"(%225, %c_10, %38) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %227 = "stablehlo.scatter"(%226, %c_10, %44) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %228 = "stablehlo.scatter"(%227, %c_0, %50) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %229 = "stablehlo.scatter"(%228, %c_0, %56) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %230 = "stablehlo.scatter"(%229, %c_0, %62) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %231 = "stablehlo.scatter"(%230, %c_1, %68) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %232 = "stablehlo.scatter"(%231, %c_1, %74) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %233 = "stablehlo.scatter"(%232, %c_1, %80) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %234 = "stablehlo.scatter"(%233, %c_2, %86) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %235 = "stablehlo.scatter"(%234, %c_2, %92) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %236 = "stablehlo.scatter"(%235, %c_2, %100) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %237 = "stablehlo.scatter"(%236, %c_3, %106) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %238 = "stablehlo.scatter"(%237, %c_3, %112) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %239 = "stablehlo.scatter"(%238, %c_3, %118) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %240 = "stablehlo.scatter"(%239, %c_4, %124) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %241 = "stablehlo.scatter"(%240, %c_4, %134) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %242 = "stablehlo.scatter"(%241, %c_4, %140) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %243 = "stablehlo.scatter"(%242, %c_5, %146) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %244 = "stablehlo.scatter"(%243, %c_5, %152) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %245 = "stablehlo.scatter"(%244, %c_5, %158) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %246 = "stablehlo.scatter"(%245, %c_6, %164) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %247 = "stablehlo.scatter"(%246, %c_6, %170) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %248 = "stablehlo.scatter"(%247, %c_6, %176) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %249 = "stablehlo.scatter"(%248, %c_8, %182) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %250 = "stablehlo.scatter"(%249, %c_8, %188) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %251 = "stablehlo.scatter"(%250, %c_8, %194) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %252 = "stablehlo.scatter"(%251, %c_9, %200) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %253 = "stablehlo.scatter"(%252, %c_9, %206) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    %254 = "stablehlo.scatter"(%253, %c_9, %212) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %255 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %255 : tensor<f32>
    }) : (tensor<1x3xf32>, tensor<1x2xi32>, tensor<1xf32>) -> tensor<1x3xf32>
    return %254 : tensor<1x3xf32>
  }
}