#unreserved__IO_FILE$def = #clift.struct<
  unique_handle = "/model-type/288",
  name = "unreserved__IO_FILE",
  size = 8,
  fields =
  [
    <
      offset = 0,
      name = "",
      type = !clift.pointer<
        pointer_size = 8,
        pointee_type = !clift.defined<
          #clift.function<
            unique_handle = "/model-type/295",
            name = "",
            return_type = !clift.primitive<VoidKind 0>,
            argument_types =
            [
              !clift.pointer<
                pointer_size = 8,
                pointee_type = !clift.defined<
                  #clift.typedef<
                    unique_handle = "/model-type/368",
                    name = "FILE_",
                    underlying_type = !clift.defined<
                      #clift.struct<unique_handle = "/model-type/288">
                    >
                  >
                >
              >
            ]
          >
        >
      >
    >
  ]
>

clift.undef : !clift.defined<#unreserved__IO_FILE$def>
