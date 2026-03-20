include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def TilingDomain_Dialect : Dialect {
  // The namespace prefix used in the IR (e.g., %0 = tiling_domain.region_op ...)
  let name = "tiling_domain";
  
  let summary = "A dialect for modeling tiling domains and region-based transformations.";
  
  let description = [{
    The `tiling_domain` dialect provides specialized operations for managing 
    tiled execution spaces, custom regions, and boundary yields. 
  }];

  // The C++ namespace where the generated C++ classes will live.
  // This helps prevent naming collisions with other dialects.
  let cppNamespace = "::mlir::tiling_domain";
  
  // Optional: If your dialect has custom types or attributes, you would 
  // enable their generation here.
  // let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;
}

class TilingDomain_Op<string mnemonic, list<Trait> traits = []> :
    Op<TilingDomain_Dialect, mnemonic, traits> {
    
  // You can define common behaviors here that ALL operations in your 
  // dialect should inherit. For most basic dialects, this class is left 
  // empty, serving strictly to bind the operations to the TilingDomain_Dialect.
}

def TilingRegionOp : TilingDomain_Op<"region_op", [
    // If your region will only ever contain one block, this trait is highly 
    // recommended. It automatically ensures the block exists and handles 
    // inserting the terminator during certain transformations.
    SingleBlockImplicitTerminator<"TilingYieldOp">,
    Pure
  ]> {
  let summary = "An operation that wraps other operations in a region";
  let description = [{
    `tiling_domain.region_op` executes the region and returns the values yielded
    by the `tiling_domain.yield` terminator.
  }];

  // The operation returns a variable number of results of any type.
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);

  // This is the crucial part: declaring the region.
  // Naming it `$body` automatically generates the `getBody()` and `getRegion()` 
  // C++ accessors.
  let regions = (region AnyRegion:$body);

  // A standard assembly format for nicely printing/parsing the IR.
  let assemblyFormat = [{
    ($inputs^ `:` type($inputs))? $body attr-dict (`:` type($results)^)?
  }];
}

def TilingYieldOp : TilingDomain_Op<"yield", [
    Terminator,
    HasParent<"TilingRegionOp">,    
    Pure,
    ReturnLike
  ]> {
  let summary = "Terminator for TilingRegionOp";
  let description = [{
    `tiling_domain.yield` yields execution and values back to the parent `region_op`.
  }];

  // It takes a variable number of operands, which correspond to the values 
  // you want the parent `TilingRegionOp` to return.
  let arguments = (ins Variadic<AnyType>:$yielded_values);

  let assemblyFormat = [{
    ($yielded_values^ `:` type($yielded_values))? attr-dict
  }];

  // Optional but helpful: A custom builder to allow creating a yield 
  // with no operands easily if the parent op returns nothing.
  let builders = [
    OpBuilder<(ins), [{ /* do nothing */ }]>
  ];
}