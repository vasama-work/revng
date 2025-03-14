!float128_t = !clift.primitive< float 16>
!generic128_t = !clift.primitive< generic 16>
!generic16_t = !clift.primitive< generic 2>
!generic32_t = !clift.primitive< generic 4>
!generic64_t = !clift.primitive< generic 8>
!generic8_t = !clift.primitive< generic 1>
!int16_t = !clift.primitive< signed 2>
!int32_t = !clift.primitive< signed 4>
!int64_t = !clift.primitive< signed 8>
!int8_t = !clift.primitive< signed 1>
!int8_t$const = !clift.primitive<const signed 1>
!number8_t$const = !clift.primitive<const number 1>
!pointer_or_number64_t = !clift.primitive< pointer_or_number 8>
!uint128_t = !clift.primitive< unsigned 16>
!uint16_t = !clift.primitive< unsigned 2>
!uint32_t = !clift.primitive< unsigned 4>
!uint64_t = !clift.primitive< unsigned 8>
!uint8_t = !clift.primitive< unsigned 1>
!uint8_t$const = !clift.primitive<const unsigned 1>
!void = !clift.primitive< void 0>
!void$const = !clift.primitive<const void 0>
#typedef_66$def = #clift.typedef<"/model-type/66" @typedef_66 : !clift.ptr< 8 -> !int8_t$const>>
!typedef_66_ = !clift.defined< #typedef_66$def>
#cabifunction_431$def = #clift.func<"/model-type/431" @cabifunction_431 : !int32_t(!typedef_66_)>
!cabifunction_431_ = !clift.defined< #cabifunction_431$def>

module {
  clift.module {
    clift.func @"local_0x401cdf:Code_x86_64"<!cabifunction_431_>(%arg0: !typedef_66_) -> !int32_t attributes {unique_handle = "/function/0x401cdf:Code_x86_64"} {
      %0 = clift.local !clift.array<216 x !generic8_t> "_local_0"
      %1 = clift.local !generic8_t "_local_1"
      %2 = clift.local !generic8_t "_local_2"
      clift.expr {
        %3 = clift.addressof %0 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>>
        %4 = clift.cast<bitcast> %3 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>> -> !generic64_t
        %5 = clift.imm 40 : !generic64_t
        %6 = clift.add %4, %5 : !generic64_t
        %7 = clift.cast<bitcast> %6 : !generic64_t -> !clift.ptr< 8 -> !void>
        %8 = clift.cast<bitcast> %arg0 : !typedef_66_ -> !generic64_t
        %9 = clift.cast<bitcast> %7 : !clift.ptr< 8 -> !void> -> !clift.ptr< 8 -> !generic64_t>
        %10 = clift.indirection %9 : < 8 -> !generic64_t>
        %11 = clift.assign %10, %8 : !generic64_t
        clift.yield %11 : !generic64_t
      }
      clift.expr {
        %3 = clift.addressof %2 : !clift.ptr< 8 -> !generic8_t>
        %4 = clift.imm 4214800 : !generic64_t
        %5 = clift.cast<bitcast> %4 : !generic64_t -> !clift.ptr< 8 -> !void>
        %6 = clift.cast<bitcast> %5 : !clift.ptr< 8 -> !void> -> !clift.ptr< 8 -> !generic64_t>
        %7 = clift.indirection %6 : < 8 -> !generic64_t>
        %8 = clift.cast<bitcast> %3 : !clift.ptr< 8 -> !generic8_t> -> !clift.ptr< 8 -> !generic64_t>
        %9 = clift.indirection %8 : < 8 -> !generic64_t>
        %10 = clift.assign %9, %7 : !generic64_t
        clift.yield %10 : !generic64_t
      }
      clift.expr {
        %3 = clift.addressof %0 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>>
        %4 = clift.cast<bitcast> %3 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>> -> !generic64_t
        %5 = clift.imm 16 : !generic64_t
        %6 = clift.add %4, %5 : !generic64_t
        %7 = clift.cast<bitcast> %6 : !generic64_t -> !clift.ptr< 8 -> !void>
        %8 = clift.addressof %0 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>>
        %9 = clift.cast<bitcast> %8 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>> -> !generic64_t
        %10 = clift.imm 224 : !generic64_t
        %11 = clift.add %9, %10 : !generic64_t
        %12 = clift.cast<bitcast> %7 : !clift.ptr< 8 -> !void> -> !clift.ptr< 8 -> !generic64_t>
        %13 = clift.indirection %12 : < 8 -> !generic64_t>
        %14 = clift.assign %13, %11 : !generic64_t
        clift.yield %14 : !generic64_t
      }
      clift.expr {
        %3 = clift.addressof %0 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>>
        %4 = clift.cast<bitcast> %3 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>> -> !generic64_t
        %5 = clift.imm 8 : !generic64_t
        %6 = clift.add %4, %5 : !generic64_t
        %7 = clift.cast<bitcast> %6 : !generic64_t -> !clift.ptr< 8 -> !void>
        %8 = clift.imm 8 : !generic32_t
        %9 = clift.cast<bitcast> %7 : !clift.ptr< 8 -> !void> -> !clift.ptr< 8 -> !generic32_t>
        %10 = clift.indirection %9 : < 8 -> !generic32_t>
        %11 = clift.assign %10, %8 : !generic32_t
        clift.yield %11 : !generic32_t
      }
      clift.expr {
        %3 = clift.addressof %0 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>>
        %4 = clift.cast<bitcast> %3 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>> -> !generic64_t
        %5 = clift.imm 12 : !generic64_t
        %6 = clift.add %4, %5 : !generic64_t
        %7 = clift.cast<bitcast> %6 : !generic64_t -> !clift.ptr< 8 -> !void>
        %8 = clift.imm 48 : !generic32_t
        %9 = clift.cast<bitcast> %7 : !clift.ptr< 8 -> !void> -> !clift.ptr< 8 -> !generic32_t>
        %10 = clift.indirection %9 : < 8 -> !generic32_t>
        %11 = clift.assign %10, %8 : !generic32_t
        clift.yield %11 : !generic32_t
      }
      clift.expr {
        %3 = clift.addressof %0 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>>
        %4 = clift.cast<bitcast> %3 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>> -> !generic64_t
        %5 = clift.imm 24 : !generic64_t
        %6 = clift.add %4, %5 : !generic64_t
        %7 = clift.cast<bitcast> %6 : !generic64_t -> !clift.ptr< 8 -> !void>
        %8 = clift.addressof %0 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>>
        %9 = clift.cast<bitcast> %8 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>> -> !generic64_t
        %10 = clift.imm 32 : !generic64_t
        %11 = clift.add %9, %10 : !generic64_t
        %12 = clift.cast<bitcast> %7 : !clift.ptr< 8 -> !void> -> !clift.ptr< 8 -> !generic64_t>
        %13 = clift.indirection %12 : < 8 -> !generic64_t>
        %14 = clift.assign %13, %11 : !generic64_t
        clift.yield %14 : !generic64_t
      }
      clift.expr {
        %3 = clift.addressof %1 : !clift.ptr< 8 -> !generic8_t>
        %4 = clift.use @"local_0x403328:Code_x86_64" : !cabifunction_433_
        %5 = clift.addressof %2 : !clift.ptr< 8 -> !generic8_t>
        %6 = clift.cast<bitcast> %5 : !clift.ptr< 8 -> !generic8_t> -> !clift.ptr< 8 -> !generic64_t>
        %7 = clift.indirection %6 : < 8 -> !generic64_t>
        %8 = clift.cast<bitcast> %7 : !generic64_t -> !typedef_88_
        %9 = clift.cast<bitcast> %arg0 : !typedef_66_ -> !generic64_t
        %10 = clift.cast<bitcast> %9 : !generic64_t -> !typedef_104_
        %11 = clift.addressof %0 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>>
        %12 = clift.cast<bitcast> %11 : !clift.ptr< 8 -> !clift.array<216 x !generic8_t>> -> !generic64_t
        %13 = clift.imm 8 : !generic64_t
        %14 = clift.add %12, %13 : !generic64_t
        %15 = clift.cast<bitcast> %14 : !generic64_t -> !clift.ptr< 8 -> !unreserved___va_list_tag>
        %16 = clift.call %4(%8, %10, %15) : !cabifunction_433_
        %17 = clift.cast<bitcast> %3 : !clift.ptr< 8 -> !generic8_t> -> !clift.ptr< 8 -> !int32_t>
        %18 = clift.indirection %17 : < 8 -> !int32_t>
        %19 = clift.assign %18, %16 : !int32_t
        clift.yield %19 : !int32_t
      }
      clift.return {
        %3 = clift.addressof %1 : !clift.ptr< 8 -> !generic8_t>
        %4 = clift.cast<bitcast> %3 : !clift.ptr< 8 -> !generic8_t> -> !clift.ptr< 8 -> !generic32_t>
        %5 = clift.indirection %4 : < 8 -> !generic32_t>
        %6 = clift.cast<bitcast> %5 : !generic32_t -> !int32_t
        clift.yield %6 : !int32_t
      }
    }
  }
}
