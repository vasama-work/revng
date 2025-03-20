//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s | %revngcliftopt

#x$def = #clift.union<
  "A" : {
    !clift.ptr<
      8 to !clift.defined<
        #clift.typedef<
          "B" : !clift.defined<#clift.union<"A">>
        >
      >
    >
  }
>

clift.undef : !clift.defined<#x$def>
