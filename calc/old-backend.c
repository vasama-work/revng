#include "types-and-globals.h"
#include "helpers.h"

_ABI(SystemV_x86_64)
generic64_t function_0x401000_Code_x86_64(generic64_t argument_0, generic64_t argument_1, union_605 *argument_2) {
  struct _PACKED struct_552 {
    uint8_t padding_at_0[8];
  } _stack;
  frame_dummy();
  unreserved___do_global_ctors_aux(argument_0, argument_1);
  return *((generic64_t *) &_stack);
}

_ABI(SystemV_x86_64) _Noreturn
void function_0x401010_Code_x86_64(void) {
  struct _PACKED struct_554 {
    generic64_t offset_0;
  } _stack;
  int32_t _var_0;
  unreserved__start_c((int64_t *) &(&_stack)[1]);
  _stack.offset_0 = (pointer_or_number64_t) &segment_0x406fd0_Generic64_2224.unreserved__data.f + 232;
  _var_0 = unreserved___libc_start_main((cabifunction_45 *) main, (int32_t) (number32_t) (&_stack)[1].offset_0, (int8_t **) &(&_stack)[2]);
  __abort("A longjmp was taken");
}

_ABI(SystemV_x86_64)
void unreserved__start_c(int64_t *p) {
  struct _PACKED struct_555 {
    uint8_t padding_at_0[8];
  } _stack;
  int32_t _var_0;
  *((generic64_t *) &_stack) = (pointer_or_number64_t) &segment_0x406fd0_Generic64_2224.unreserved__data.f + 232;
  _var_0 = unreserved___libc_start_main((cabifunction_45 *) main, (int32_t) (number32_t) *p, (int8_t **) &p[1]);
}

_ABI(SystemV_x86_64)
void deregister_tm_clones(void) {
}

_ABI(SystemV_x86_64)
void register_tm_clones(void) {
}

_ABI(SystemV_x86_64)
void unreserved___do_global_dtors_aux(void) {
  struct _PACKED struct_557 {
    uint8_t padding_at_0[8];
    generic64_t offset_8;
    uint8_t padding_at_16[8];
  } _stack;
  if (!segment_0x406fd0_Generic64_2224.unreserved__bss.completed_5933) {
    _stack.offset_8 = &segment_0x406fd0_Generic64_2224.unreserved__dtors;
    deregister_tm_clones();
    segment_0x406fd0_Generic64_2224.unreserved__bss.completed_5933 = '\001';
  }
}

_ABI(SystemV_x86_64)
void frame_dummy(void) {
}

_ABI(SystemV_x86_64) _Noreturn
void function_0x401171_Code_x86_64(void) {
  __abort("A longjmp was taken");
}

_ABI(SystemV_x86_64)
int32_t root(int8_t *buffer, size_t size) {
  struct _PACKED struct_558 {
    generic64_t offset_0;
    generic64_t offset_8;
    uint8_t padding_at_16[204];
    generic32_t offset_220;
    generic32_t offset_224;
    generic32_t offset_228;
    generic32_t offset_232;
    generic32_t offset_236;
    uint8_t padding_at_240[8];
  } _stack;
  uint64_t _loop_state_var;
  generic32_t _var_0;
  _stack.offset_8 = buffer;
  _stack.offset_0 = size;
  _stack.offset_236 = 4294967295;
  _stack.offset_232 = 0;
  _stack.offset_228 = 0;
  _stack.offset_224 = 0;
  if (_stack.offset_0 > (uint64_t) _stack.offset_232) {
    generic64_t _var_1;
    _var_1 = _stack.offset_232;
    while (true) {
      generic32_t _var_2;
      bool _break_from_loop_3 = false;
      switch ((number8_t) *((generic8_t *) (_stack.offset_8 + _var_1))) {
        case 40:
        {
          if (_stack.offset_224) {
            _var_0 = 666;
            _loop_state_var = 0;
            _break_from_loop_3 = true;
            break;
          }
          generic32_t _var_4;
          _stack.offset_228 = 0;
          _var_4 = _stack.offset_236;
          _stack.offset_236 = _var_4 + 1;
          if (_var_4 == 9) {
            _var_0 = 666;
            _loop_state_var = 0;
            _break_from_loop_3 = true;
            break;
          }
          generic32_t _var_5;
          _stack.offset_220 = 0;
          _var_5 = 0;
          generic32_t _var_6;
          do {
            *((generic8_t *) ((pointer_or_number64_t) &_stack.offset_8 + 8 + (number64_t) _stack.offset_236 * 20 + _var_5)) = '\000';
            _var_6 = _stack.offset_220;
            _var_5 = _var_6 + 1;
            _stack.offset_220 = _var_5;
          } while (!(_var_6 > 18 && _var_6 < (uint32_t) -1));
          _stack.offset_232 = _stack.offset_232 + 1;
          if (_stack.offset_0 == (pointer_or_number64_t) _stack.offset_232) {
            _var_0 = 666;
            _loop_state_var = 0;
            _break_from_loop_3 = true;
            break;
          }
          generic16_t _var_7;
          _var_7 = 1;
          switch ((number8_t) *((generic8_t *) (_stack.offset_8 + (pointer_or_number64_t) _stack.offset_232))) {
            case 33:
            case 38:
            case 42:
            case 43:
            case 45:
            case 63:
            case 94:
            case 124:
            case 126:
            {
              break;
            } break;
            default:
            {
              _var_0 = 666;
              _loop_state_var = 0;
              _break_from_loop_3 = true;
              break;
            } break;
          }
          if (_break_from_loop_3)
            break;
          switch ((number8_t) *((generic8_t *) (_stack.offset_8 + (pointer_or_number64_t) _stack.offset_232))) {
            case 38:
            case 42:
            case 43:
            case 45:
            case 94:
            case 124:
            {
              _var_7 = 2;
            } break;
            case 63:
            {
              _var_7 = 3;
            } break;
          }
          *((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 222)) = _var_7;
          *((generic8_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 224)) = *((generic8_t *) (_stack.offset_8 + (pointer_or_number64_t) _stack.offset_232));
          *((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 220)) = 0;
          _stack.offset_232 = _stack.offset_232 + 1;
          if (_stack.offset_0 == (pointer_or_number64_t) _stack.offset_232) {
            _var_0 = 666;
            _loop_state_var = 0;
            _break_from_loop_3 = true;
            break;
          }
          if (*((generic8_t *) (_stack.offset_8 + (pointer_or_number64_t) _stack.offset_232)) != ' ') {
            _var_0 = 666;
            _loop_state_var = 0;
            _break_from_loop_3 = true;
            break;
          }
        } break;
        case 45:
        {
          _var_2 = 2;
          if (_stack.offset_224) {
            _var_0 = 666;
            _loop_state_var = 0;
            _break_from_loop_3 = true;
            break;
          }
          _stack.offset_224 = _var_2;
        } break;
        default:
        {
          if (*((generic8_t *) (_stack.offset_8 + _var_1)) < '0' || *((generic8_t *) (_stack.offset_8 + _var_1)) > '9') {
            switch ((number8_t) *((generic8_t *) (_stack.offset_8 + _var_1))) {
              case 32:
              {
                if (_stack.offset_236 == (pointer_or_number32_t) -1) {
                  _var_0 = 666;
                  _loop_state_var = 0;
                  _break_from_loop_3 = true;
                  break;
                }
                *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + ((number64_t) _stack.offset_236 * 5 + *((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 220))) * 4 - 216)) = _stack.offset_228;
                _stack.offset_224 = 0;
                _stack.offset_228 = 0;
                *((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 220)) = *((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 220)) + 1;
                if (*((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 220)) == *((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 222))) {
                  _var_0 = 666;
                  _loop_state_var = 0;
                  _break_from_loop_3 = true;
                  break;
                }
              } break;
              case 41:
              {
                if (_stack.offset_236 == (pointer_or_number32_t) -1) {
                  _var_0 = 666;
                  _loop_state_var = 0;
                  _break_from_loop_3 = true;
                  break;
                }
                *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + ((number64_t) _stack.offset_236 * 5 + *((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 220))) * 4 - 216)) = _stack.offset_228;
                _stack.offset_224 = 0;
                _stack.offset_228 = 0;
                if (*((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 220)) + 1 != *((generic16_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 222))) {
                  _var_0 = 666;
                  _loop_state_var = 0;
                  _break_from_loop_3 = true;
                  break;
                }
                switch ((number8_t) *((generic8_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 224))) {
                  case 33:
                  case 38:
                  case 42:
                  case 43:
                  case 45:
                  case 63:
                  case 94:
                  case 124:
                  case 126:
                  {
                    break;
                  } break;
                  default:
                  {
                    _var_0 = 666;
                    _loop_state_var = 0;
                    _break_from_loop_3 = true;
                    break;
                  } break;
                }
                if (_break_from_loop_3)
                  break;
                generic32_t _var_8;
                switch ((number8_t) *((generic8_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 224))) {
                  case 43:
                  {
                    _var_8 = *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216)) + *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 212));
                  } break;
                  case 45:
                  {
                    _var_8 = *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216)) - *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 212));
                  } break;
                  case 42:
                  {
                    _var_8 = *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 212)) * *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216));
                  } break;
                  case 38:
                  {
                    _var_8 = *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216)) & *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 212));
                  } break;
                  case 124:
                  {
                    _var_8 = *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216)) | *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 212));
                  } break;
                  case 94:
                  {
                    _var_8 = *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216)) ^ *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 212));
                  } break;
                  case 63:
                  {
                    generic64_t _var_9;
                    _var_9 = !*((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216)) ? 18446744073709551408U : 18446744073709551404U;
                    _var_8 = *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 + _var_9 * 1));
                  } break;
                  case 126:
                  {
                    _var_8 = ~*((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216));
                  } break;
                  case 33:
                  {
                    _var_8 = !*((generic32_t *) ((pointer_or_number64_t) &_stack.offset_236 + 4 + (number64_t) _stack.offset_236 * 20 - 216));
                  } break;
                }
                generic32_t _var_10;
                _stack.offset_228 = _var_8;
                _var_10 = _stack.offset_236;
                _stack.offset_236 = _var_10 - 1;
                if (_var_10 > (uint32_t) -2147483648) {
                  _var_0 = 666;
                  _loop_state_var = 0;
                  _break_from_loop_3 = true;
                  break;
                }
              } break;
            }
            if (_break_from_loop_3)
              break;
          } else {
            _stack.offset_228 = (pointer_or_number32_t) *((generic8_t *) (_stack.offset_8 + _var_1)) - 48 + _stack.offset_228 * 10;
            _var_2 = 1;
            if (_stack.offset_224 == 2) {
              _stack.offset_228 = 0 - _stack.offset_228;
              _var_2 = 1;
            }
            _stack.offset_224 = _var_2;
          }
        } break;
      }
      if (_break_from_loop_3)
        break;
      _stack.offset_232 = _stack.offset_232 + 1;
      _var_1 = _stack.offset_232;
      if (_stack.offset_0 > _var_1) {
        continue;
      }
      break;
    }
    if (!(_loop_state_var)) {
      return (int32_t) _var_0;
    }
  }
  _var_0 = _stack.offset_228;
  return (int32_t) _var_0;
}

_ABI(SystemV_x86_64)
int32_t main(int32_t argc, int8_t **argv) {
  struct _PACKED struct_559 {
    struct_686 *offset_0;
    uint8_t padding_at_8[4];
    generic32_t offset_12;
    uint8_t padding_at_16[8];
  } _stack;
  int32_t _var_0;
  int32_t _var_1;
  size_t _var_2;
  _stack.offset_12 = argc;
  _stack.offset_0 = argv;
  _var_2 = strlen(argv[1]);
  _var_1 = root((int8_t *) &_stack.offset_0->offset_8->offset_0.member_0.offset_0, _var_2);
  _var_0 = printf((typedef_66) "%d\n");
  return (int32_t) 0;
}

_ABI(SystemV_x86_64)
void dummy(void) {
}

_ABI(SystemV_x86_64)
void dummy1(void *p) {
}

_ABI(SystemV_x86_64)
void unreserved___init_libc(int8_t **envp, int8_t *pn) {
  struct _PACKED struct_561 {
    union_632 offset_0;
    uint8_t padding_at_340[4];
  } _stack;
  uint64_t _loop_state_var;
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  generic32_t _var_24;
  generic64_t _var_25;
  generic64_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic32_t _var_29;
  generic64_t _var_30;
  generic32_t _var_31;
  generic32_t _var_32;
  generic32_t _var_33;
  generic64_t _var_34;
  generic32_t _var_35;
  generic32_t _var_36;
  generic32_t _var_37;
  generic64_t _var_38;
  generic32_t _var_39;
  generic32_t _var_40;
  generic64_t _var_41;
  generic32_t _var_42;
  generic32_t _var_43;
  generic32_t _var_44;
  generic64_t _var_45;
  generic32_t _var_46;
  generic8_t _var_47;
  generic64_t _var_48;
  generic64_t _var_49;
  segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved__environ = envp;
  _var_49 = &_stack.offset_0.member_0.offset_32;
  _var_48 = 0;
  do {
    *((generic32_t *) _var_49) = 0;
    _var_49 = _var_49 + 4;
    _var_48 = _var_48 + 1;
  } while (_var_48 != 76);
  generic64_t _var_50;
  _var_50 = 0;
  generic64_t _var_51;
  do {
    _var_51 = _var_50;
    _var_50 = _var_51 + 1;
  } while (envp[_var_51]);
  segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_16 = &envp[_var_51 + 1];
  if (envp[_var_51 + 1]) {
    generic64_t _var_52;
    int8_t *_var_53;
    _var_52 = 0;
    _var_53 = envp[_var_51 + 1];
    do {
      if (!((uint64_t) _var_53 > 37)) {
        *((int8_t **) &_stack.offset_0.member_0.offset_32.member_12[2 * (number64_t) _var_53]) = envp[_var_51 + (2 * _var_52 + 2)];
      }
      _var_53 = envp[_var_51 + (2 * _var_52 + 3)];
      _var_52 = _var_52 + 1;
    } while (_var_53);
  }
  segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___hwcap = _stack.offset_0.member_0.offset_32.member_0.offset_128;
  segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___sysinfo = _stack.offset_0.member_0.offset_32.member_1.offset_256;
  segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_32 = _stack.offset_0.member_0.offset_32.member_2.offset_48;
  if (pn) {
    int8_t *_var_54;
    generic64_t _var_55;
    segment_0x406fd0_Generic64_2224.unreserved__bss.program_invocation_name = pn;
    _var_54 = pn;
    _var_55 = _stack.offset_0.member_0.offset_32.member_2.offset_48;
    while (true) {
      generic64_t _var_56;
      generic64_t _var_57;
      generic64_t _var_58;
      _var_57 = _var_54;
      _var_58 = _var_55;
      segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___progname = _var_57;
      _var_56 = 0;
      generic64_t _var_59;
      while (true) {
        _var_59 = _var_56;
        if (*((generic8_t *) _var_57)) {
          generic8_t _var_60;
          _var_58 = (_var_58 & 0xFFFFFFFFFFFFFF00) | *((generic8_t *) _var_57);
          _var_60 = *((generic8_t *) _var_57) == '/';
          _var_56 = _var_59 + 1;
          _var_57 = &((int8_t *) _var_57)[1];
          if (!(_var_60)) {
            continue;
          }
          _loop_state_var = 1;
          break;
        }
        _loop_state_var = 0;
        break;
      }
      if (!(_loop_state_var)) {
        break;
      }
      _var_54 = &_var_54[_var_59 + 1];
      _var_55 = _var_58;
    }
  }
  unreserved___init_tls((size_t *) &_stack.offset_0.member_0.offset_32);
  dummy1((void *) _stack.offset_0.member_0.offset_32.member_3.offset_200);
  if (((_stack.offset_0.member_0.offset_32.member_5.offset_88 == _stack.offset_0.member_0.offset_32.member_4.offset_96) && (_stack.offset_0.member_0.offset_32.member_7.offset_104 == _stack.offset_0.member_0.offset_32.member_6.offset_112)) && (!_stack.offset_0.member_0.offset_32.member_8.offset_184)) {
    return;
  }
  generic64_t _var_61;
  generic64_t _var_62;
  _var_62 = &_stack.offset_0.member_3.offset_8;
  _var_61 = 0;
  do {
    *((generic32_t *) _var_62) = 0;
    _var_62 = _var_62 + 4;
    _var_61 = _var_61 + 1;
  } while (_var_61 != 6);
  generic32_t _var_63;
  generic64_t _var_64;
  generic32_t _var_65;
  generic64_t _var_66;
  generic64_t _var_67;
  generic64_t _var_68;
  generic64_t _var_69;
  generic32_t _var_70;
  generic32_t _var_71;
  generic64_t _var_72;
  generic32_t _var_73;
  _stack.offset_0.member_3.offset_8[2] = 1;
  _stack.offset_0.member_3.offset_8[4] = 2;
  _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 3092, _undef_generic64_t(), (pointer_or_number64_t) &_stack + 8, _undef_generic64_t(), 7, _undef_generic64_t(), 0, 0, (pointer_or_number64_t) &_stack + 8, 0, 3, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32, &_var_33, &_var_34, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42, &_var_43, &_var_44, &_var_45, &_var_46, &_var_47);
  _var_63 = _var_29;
  _var_64 = _var_30;
  _var_65 = _var_32;
  _var_67 = _var_34;
  _var_68 = _var_38;
  _var_69 = _var_41;
  _var_70 = _var_42;
  _var_71 = _var_43;
  _var_72 = _var_45;
  _var_73 = _var_46;
  _var_66 = 0;
  do {
    generic32_t _var_74;
    generic64_t _var_75;
    generic32_t _var_76;
    generic64_t _var_77;
    generic64_t _var_78;
    generic64_t _var_79;
    generic32_t _var_80;
    generic32_t _var_81;
    generic64_t _var_82;
    generic32_t _var_83;
    _var_74 = _var_63;
    _var_75 = _var_64;
    _var_76 = _var_65;
    _var_77 = _var_67;
    _var_78 = _var_68;
    _var_79 = _var_69;
    _var_80 = _var_70;
    _var_81 = _var_71;
    _var_82 = _var_72;
    _var_83 = _var_73;
    if ((_stack.offset_0.member_1.offset_14.offset_0[_var_66].offset_0 & 0x20)) {
      _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 3115, _undef_generic64_t(), (pointer_or_number64_t) &_stack + 8, _undef_generic64_t(), 2, _undef_generic64_t(), _var_66, 0, "/dev/null", 0, 2, _var_63, _var_64, _var_65, 0, 0, 15727360, 0, 13628160, 0, _var_67, _var_68, _var_69, _var_70, 274877906944, 127, 2147549185, 0, _var_71, _var_72, _var_73, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
      _var_74 = _var_5;
      _var_75 = _var_6;
      _var_76 = _var_8;
      _var_77 = _var_10;
      _var_78 = _var_14;
      _var_79 = _var_17;
      _var_80 = _var_18;
      _var_81 = _var_19;
      _var_82 = _var_21;
      _var_83 = _var_22;
    }
    _var_66 = _var_66 + 1;
  } while (_var_66 != 3);
  segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_8 = 1;
}

_ABI(SystemV_x86_64)
int32_t unreserved___libc_start_main(cabifunction_45 *main_, int32_t argc, int8_t **argv) {
  pointer_or_number64_t _var_0;
  pointer_or_number64_t _var_1;
  generic64_t _var_2;
  artificial_struct_returned_by_rawfunction_25 _var_3;
  unreserved___init_libc((int8_t **) ((pointer_or_number64_t) &argv[1] + ((int64_t) ((number64_t) (uint64_t) argc << 32) >> 29) * 1), *argv);
  _var_2 = function_0x401000_Code_x86_64((pointer_or_number64_t) &argv[1] + ((int64_t) ((number64_t) (uint64_t) argc << 32) >> 29) * 1, (generic64_t) *argv, (union_605 *) argv);
  _var_3 = ((rawfunction_25 *) main_)(_undef_generic64_t(), (pointer_or_number64_t) &argv[1] + ((int64_t) ((number64_t) (uint64_t) argc << 32) >> 29) * 1, (pointer_or_number64_t) argv, (uint64_t) argc, _undef_generic64_t(), _undef_generic64_t());
  _var_1 = _var_3.register_rax;
  _var_0 = _var_3.register_rdx;
  exit((int32_t) (number32_t) _var_1);
  // The previous function call does not return
}

_ABI(SystemV_x86_64)
void dummy_(void) {
}

_ABI(SystemV_x86_64) _Noreturn
void exit(int32_t code) {
  struct_755 _var_0;
  dummy_();
  _var_0 = function_0x404091_Code_x86_64();
  unreserved___stdio_exit();
  unreserved__Exit(code);
  // The previous function call does not return
}

_ABI(SystemV_x86_64)
int32_t printf(typedef_66 fmt) {
  struct _PACKED struct_560 {
    uint8_t padding_at_0[8];
    struct_616 offset_8;
    uint8_t padding_at_32[8];
    union_605 *offset_40;
    uint8_t padding_at_48[168];
  } _stack;
  int32_t _var_0;
  _stack.offset_40 = fmt;
  _stack.offset_8.offset_8 = &(&_stack)[1].offset_8;
  _stack.offset_8.offset_0.member_1 = 8;
  _stack.offset_8.offset_0.member_0.offset_4 = 48;
  _stack.offset_8.offset_16 = (pointer_or_number64_t) &_stack.offset_8.offset_16 + 8;
  _var_0 = vfprintf((typedef_88) *((generic64_t *) "`p@"), fmt, (unreserved___va_list_tag *) &_stack.offset_8);
  return _var_0;
}

_ABI(SystemV_x86_64)
int8_t *fmt_u(unreserved_uintmax_t x, int8_t *s) {
  generic64_t _var_0;
  _var_0 = s;
  if (x) {
    generic64_t _var_1;
    generic64_t _var_2;
    _var_1 = 0;
    _var_2 = x;
    generic64_t _var_3;
    generic64_t _var_4;
    do {
      _var_3 = _var_2;
      _var_4 = (pointer_or_number64_t) s - 1 - _var_1;
      _var_2 = _var_3 / 10;
      *((generic8_t *) _var_4) = (number8_t) (_var_3 % 10) | 0x30;
      _var_1 = _var_1 + 1;
    } while (!(_var_3 < 10));
    _var_0 = _var_4;
  }
  return (int8_t *) _var_0;
}

_ABI(SystemV_x86_64)
void out(FILE_ *f, const int8_t *s, size_t l) {
  struct _PACKED struct_567 {
    struct_724 *offset_0;
    uint8_t padding_at_8[32];
  } _stack;
  if (!(*((generic8_t *) f) & 0x20)) {
    _stack.offset_0 = f;
    if (!f->wend) {
      int32_t _var_0;
      _var_0 = unreserved___towrite(f);
      if (_var_0) {
        return;
      }
    }
    if (!((pointer_or_number64_t) f->wend - (number64_t) f->wpos < l)) {
      generic64_t _var_1;
      generic64_t _var_2;
      _var_1 = s;
      _var_2 = l;
      if (!(f->lbf < (int8_t) 0 || !l)) {
        generic64_t _var_3;
        generic64_t _var_4;
        _var_3 = 0;
        _var_4 = l;
        while (true) {
          if ((pointer_or_number8_t) s[~_var_3 + l] == '\n') {
            pointer_or_number64_t _var_5;
            pointer_or_number64_t _var_6;
            artificial_struct_returned_by_rawfunction_25 _var_7;
            _var_7 = ((rawfunction_25 *) f->write)(_undef_generic64_t(), _var_4, (pointer_or_number64_t) s, (pointer_or_number64_t) f, (pointer_or_number64_t) f, _undef_generic64_t());
            _var_6 = _var_7.register_rax;
            _var_5 = _var_7.register_rdx;
            if (_var_4 > _var_6) {
              return;
            }
            _var_2 = l - _var_4;
            _var_1 = (pointer_or_number64_t) &s[l] - _var_3;
          } else {
            generic8_t _var_8;
            _var_4 = _var_4 - 1;
            _var_8 = ~_var_3 == 0 - l;
            _var_3 = _var_3 + 1;
            if (!(_var_8)) {
              continue;
            }
            _var_1 = s;
            _var_2 = l;
          }
          break;
        }
      }
      struct_718 *_var_9;
      _var_9 = memcpy((struct_718 *) f->wpos, (union_596 *) _var_1, _var_2);
      f->wpos = &f->wpos[_var_2];
    }
  }
}

_ABI(SystemV_x86_64)
void pop_arg(arg *arg_, int32_t type, va_list *ap) {
  generic32_t _var_0;
  generic8_t _var_1;
  generic8_t _var_2;
  generic8_t _var_3;
  generic8_t _var_4;
  generic8_t _var_5;
  generic8_t _var_6;
  generic8_t _var_7;
  generic8_t _var_8;
  generic32_t _var_9;
  generic8_t _var_10;
  generic8_t _var_11;
  generic8_t _var_12;
  generic8_t _var_13;
  generic8_t _var_14;
  generic8_t _var_15;
  generic8_t _var_16;
  generic8_t _var_17;
  generic64_t _var_18;
  generic16_t _var_19;
  generic64_t _var_20;
  generic16_t _var_21;
  generic64_t _var_22;
  generic16_t _var_23;
  generic64_t _var_24;
  generic16_t _var_25;
  generic64_t _var_26;
  generic16_t _var_27;
  generic64_t _var_28;
  generic16_t _var_29;
  generic64_t _var_30;
  generic16_t _var_31;
  generic64_t _var_32;
  generic16_t _var_33;
  generic8_t _var_34;
  generic32_t _var_35;
  generic8_t _var_36;
  generic8_t _var_37;
  generic8_t _var_38;
  generic8_t _var_39;
  generic8_t _var_40;
  generic8_t _var_41;
  generic8_t _var_42;
  generic8_t _var_43;
  generic32_t _var_44;
  generic8_t _var_45;
  generic8_t _var_46;
  generic8_t _var_47;
  generic8_t _var_48;
  generic8_t _var_49;
  generic8_t _var_50;
  generic8_t _var_51;
  generic8_t _var_52;
  generic64_t _var_53;
  generic16_t _var_54;
  generic64_t _var_55;
  generic16_t _var_56;
  generic64_t _var_57;
  generic16_t _var_58;
  generic64_t _var_59;
  generic16_t _var_60;
  generic64_t _var_61;
  generic16_t _var_62;
  generic64_t _var_63;
  generic16_t _var_64;
  generic64_t _var_65;
  generic16_t _var_66;
  generic64_t _var_67;
  generic16_t _var_68;
  if ((!((uint32_t) type > 22)) && (!((uint32_t) type > 20 || (uint32_t) type < 9))) {
    switch ((number64_t) *((generic64_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.stdout + 16 + ((pointer_or_number32_t) type - 9) * 8))) {
      case 4202215:
      {
        generic64_t _var_69;
        if ((*ap)[0].fp_offset > 175) {
          _var_69 = (*ap)[0].overflow_arg_area;
          (*ap)[0].overflow_arg_area = _var_69 + 8;
        } else {
          _var_69 = (pointer_or_number64_t) (*ap)[0].reg_save_area + (*ap)[0].fp_offset;
          (*ap)[0].fp_offset = (*ap)[0].fp_offset + 16;
        }
        _helper_fldl_ST0_wrapper(NULL, *((generic64_t *) _var_69), 0, '\000', '\000', '\000', &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32, &_var_33, &_var_34);
        _helper_fstt_ST0_wrapper(NULL, arg_, _var_9, _var_18, _var_19, _var_20, _var_21, _var_22, _var_23, _var_24, _var_25, _var_26, _var_27, _var_28, _var_29, _var_30, _var_31, _var_32, _var_33);
        _helper_fpop_wrapper(NULL, _var_9, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8);
      } break;
      case 4202257:
      {
        (*ap)[0].overflow_arg_area = (((pointer_or_number64_t) (*ap)[0].overflow_arg_area + 15) & 0xFFFFFFFFFFFFFFF0) + 16;
        _helper_fldt_ST0_wrapper(NULL, ((pointer_or_number64_t) (*ap)[0].overflow_arg_area + 15) & 0xFFFFFFFFFFFFFFF0, 0, &_var_44, &_var_45, &_var_46, &_var_47, &_var_48, &_var_49, &_var_50, &_var_51, &_var_52, &_var_53, &_var_54, &_var_55, &_var_56, &_var_57, &_var_58, &_var_59, &_var_60, &_var_61, &_var_62, &_var_63, &_var_64, &_var_65, &_var_66, &_var_67, &_var_68);
        _helper_fstt_ST0_wrapper(NULL, arg_, _var_44, _var_53, _var_54, _var_55, _var_56, _var_57, _var_58, _var_59, _var_60, _var_61, _var_62, _var_63, _var_64, _var_65, _var_66, _var_67, _var_68);
        _helper_fpop_wrapper(NULL, _var_44, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42, &_var_43);
      } break;
      case 4201944:
      case 4201983:
      case 4202019:
      case 4202056:
      case 4202094:
      case 4202134:
      case 4202175:
      {
        generic64_t _var_70;
        switch ((number64_t) *((generic64_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.stdout + 16 + ((pointer_or_number32_t) type - 9) * 8))) {
          case 4202056:
          {
            generic64_t _var_71;
            if ((*ap)[0].gp_offset > 47) {
              _var_71 = (*ap)[0].overflow_arg_area;
              (*ap)[0].overflow_arg_area = _var_71 + 8;
            } else {
              _var_71 = (pointer_or_number64_t) (*ap)[0].reg_save_area + (*ap)[0].gp_offset;
              *((generic32_t *) ap) = (*ap)[0].gp_offset + 8;
            }
            _var_70 = *((generic16_t *) _var_71);
          } break;
          case 4202175:
          {
            generic64_t _var_72;
            if ((*ap)[0].gp_offset > 47) {
              _var_72 = (*ap)[0].overflow_arg_area;
              (*ap)[0].overflow_arg_area = _var_72 + 8;
            } else {
              _var_72 = (pointer_or_number64_t) (*ap)[0].reg_save_area + (*ap)[0].gp_offset;
              *((generic32_t *) ap) = (*ap)[0].gp_offset + 8;
            }
            _var_70 = *((generic8_t *) _var_72);
          } break;
          case 4201944:
          {
            generic64_t _var_73;
            if ((*ap)[0].gp_offset > 47) {
              _var_73 = (*ap)[0].overflow_arg_area;
              (*ap)[0].overflow_arg_area = _var_73 + 8;
            } else {
              _var_73 = (pointer_or_number64_t) (*ap)[0].reg_save_area + (*ap)[0].gp_offset;
              *((generic32_t *) ap) = (*ap)[0].gp_offset + 8;
            }
            _var_70 = *((generic32_t *) _var_73);
          } break;
          case 4202019:
          {
            generic64_t _var_74;
            if ((*ap)[0].gp_offset > 47) {
              _var_74 = (*ap)[0].overflow_arg_area;
              (*ap)[0].overflow_arg_area = _var_74 + 8;
            } else {
              _var_74 = (pointer_or_number64_t) (*ap)[0].reg_save_area + (*ap)[0].gp_offset;
              *((generic32_t *) ap) = (*ap)[0].gp_offset + 8;
            }
            _var_70 = *((generic64_t *) _var_74);
          } break;
          case 4202094:
          {
            generic64_t _var_75;
            if ((*ap)[0].gp_offset > 47) {
              _var_75 = (*ap)[0].overflow_arg_area;
              (*ap)[0].overflow_arg_area = _var_75 + 8;
            } else {
              _var_75 = (pointer_or_number64_t) (*ap)[0].reg_save_area + (*ap)[0].gp_offset;
              *((generic32_t *) ap) = (*ap)[0].gp_offset + 8;
            }
            _var_70 = *((generic16_t *) _var_75);
          } break;
          case 4201983:
          {
            generic64_t _var_76;
            if ((*ap)[0].gp_offset > 47) {
              _var_76 = (*ap)[0].overflow_arg_area;
              (*ap)[0].overflow_arg_area = _var_76 + 8;
            } else {
              _var_76 = (pointer_or_number64_t) (*ap)[0].reg_save_area + (*ap)[0].gp_offset;
              *((generic32_t *) ap) = (*ap)[0].gp_offset + 8;
            }
            _var_70 = *((generic32_t *) _var_76);
          } break;
          case 4202134:
          {
            generic64_t _var_77;
            if ((*ap)[0].gp_offset > 47) {
              _var_77 = (*ap)[0].overflow_arg_area;
              (*ap)[0].overflow_arg_area = _var_77 + 8;
            } else {
              _var_77 = (pointer_or_number64_t) (*ap)[0].reg_save_area + (*ap)[0].gp_offset;
              *((generic32_t *) ap) = (*ap)[0].gp_offset + 8;
            }
            _var_70 = *((generic8_t *) _var_77);
          } break;
        }
        arg_->i = _var_70;
      } break;
      default:
      {
      } break;
    }
  }
}

_ABI(SystemV_x86_64)
void pad(FILE_ *f, int8_t c, int32_t w, int32_t l, int32_t fl) {
  struct _PACKED struct_568 {
    struct_661 offset_0;
    uint8_t padding_at_63[217];
  } _stack;
  if ((!((number32_t) fl & 0x12000)) && ((int64_t) ((number64_t) (uint64_t) l << 32) < (int64_t) ((number64_t) (uint64_t) w << 32))) {
    struct_661 *_var_0;
    generic64_t _var_1;
    _var_1 = (int32_t) (number32_t) ((uint64_t) w - (uint64_t) l) > (int32_t) 256 ? 256 : (int64_t) ((number64_t) ((uint64_t) w - (uint64_t) l) << 32) >> 32;
    _var_0 = memset(&_stack.offset_0, (number64_t) c & 0xFFFFFFFF, _var_1, (uint64_t) l, 0);
    if ((int32_t) (number32_t) ((uint64_t) w - (uint64_t) l) > (int32_t) 255) {
      generic32_t _var_2;
      _var_2 = 0;
      generic8_t _var_3;
      do {
        out(f, (const int8_t *) &_stack.offset_0.offset_0.member_0.member_0.member_1, 256);
        _var_3 = (int32_t) ((pointer_or_number32_t) w - 256 - (((number32_t) _var_2 << 8) + (pointer_or_number32_t) l)) > (int32_t) 255;
        _var_2 = _var_2 + 1;
      } while (_var_3);
    }
    out(f, (const int8_t *) &_stack.offset_0.offset_0.member_0.member_0.member_1, (int64_t) ((number64_t) ((((uint64_t) ((uint64_t) w - (uint64_t) l) >> 8) & 0xFFFFFF) * 4294967040 + ((uint64_t) w - (uint64_t) l)) << 32) >> 32);
  }
}

_ABI(SystemV_x86_64)
int32_t fmt_fp(FILE_ *f, float128_t y, int32_t w, int32_t p, int32_t fl, int32_t t) {
  struct _PACKED struct_569 {
    generic64_t offset_0;
    generic64_t offset_8;
    generic32_t offset_16;
    generic8_t offset_20;
    uint8_t padding_at_21[3];
    union_674 offset_24;
    union_675 offset_32;
    uint8_t padding_at_40[8];
    union_676 offset_48;
    generic32_t offset_56;
    uint8_t padding_at_60[4];
    generic32_t offset_64;
    uint8_t padding_at_68[4];
    generic32_t offset_72;
    generic16_t offset_76;
    generic16_t offset_78;
    uint8_t padding_at_80[8];
    generic32_t offset_88;
    uint8_t padding_at_92[14];
    generic8_t offset_106;
    union_677 offset_107;
    uint8_t padding_at_148[7380];
  } _stack;
  int32_t _var_0;
  generic64_t _var_1;
  generic16_t _var_2;
  generic64_t _var_3;
  generic16_t _var_4;
  generic64_t _var_5;
  generic16_t _var_6;
  generic64_t _var_7;
  generic16_t _var_8;
  generic64_t _var_9;
  generic16_t _var_10;
  generic64_t _var_11;
  generic16_t _var_12;
  generic64_t _var_13;
  generic16_t _var_14;
  generic64_t _var_15;
  generic16_t _var_16;
  generic64_t _var_17;
  generic16_t _var_18;
  generic64_t _var_19;
  generic16_t _var_20;
  generic64_t _var_21;
  generic16_t _var_22;
  generic64_t _var_23;
  generic16_t _var_24;
  generic64_t _var_25;
  generic16_t _var_26;
  generic64_t _var_27;
  generic16_t _var_28;
  generic64_t _var_29;
  generic16_t _var_30;
  generic64_t _var_31;
  generic16_t _var_32;
  generic64_t _var_33;
  generic16_t _var_34;
  generic64_t _var_35;
  generic16_t _var_36;
  generic64_t _var_37;
  generic16_t _var_38;
  generic64_t _var_39;
  generic16_t _var_40;
  generic64_t _var_41;
  generic16_t _var_42;
  generic64_t _var_43;
  generic16_t _var_44;
  generic64_t _var_45;
  generic16_t _var_46;
  generic64_t _var_47;
  generic16_t _var_48;
  generic64_t _var_49;
  generic16_t _var_50;
  generic64_t _var_51;
  generic16_t _var_52;
  generic64_t _var_53;
  generic16_t _var_54;
  generic64_t _var_55;
  generic16_t _var_56;
  generic64_t _var_57;
  generic16_t _var_58;
  generic64_t _var_59;
  generic16_t _var_60;
  generic64_t _var_61;
  generic16_t _var_62;
  generic64_t _var_63;
  generic16_t _var_64;
  generic32_t _var_65;
  generic8_t _var_66;
  generic8_t _var_67;
  generic8_t _var_68;
  generic8_t _var_69;
  generic8_t _var_70;
  generic8_t _var_71;
  generic8_t _var_72;
  generic8_t _var_73;
  generic64_t _var_74;
  generic16_t _var_75;
  generic64_t _var_76;
  generic16_t _var_77;
  generic64_t _var_78;
  generic16_t _var_79;
  generic64_t _var_80;
  generic16_t _var_81;
  generic64_t _var_82;
  generic16_t _var_83;
  generic64_t _var_84;
  generic16_t _var_85;
  generic64_t _var_86;
  generic16_t _var_87;
  generic64_t _var_88;
  generic16_t _var_89;
  generic32_t _var_90;
  generic8_t _var_91;
  generic8_t _var_92;
  generic8_t _var_93;
  generic8_t _var_94;
  generic8_t _var_95;
  generic8_t _var_96;
  generic8_t _var_97;
  generic8_t _var_98;
  generic64_t _var_99;
  generic16_t _var_100;
  generic64_t _var_101;
  generic16_t _var_102;
  generic64_t _var_103;
  generic16_t _var_104;
  generic64_t _var_105;
  generic16_t _var_106;
  generic64_t _var_107;
  generic16_t _var_108;
  generic64_t _var_109;
  generic16_t _var_110;
  generic64_t _var_111;
  generic16_t _var_112;
  generic64_t _var_113;
  generic16_t _var_114;
  generic32_t _var_115;
  generic8_t _var_116;
  generic8_t _var_117;
  generic8_t _var_118;
  generic8_t _var_119;
  generic8_t _var_120;
  generic8_t _var_121;
  generic8_t _var_122;
  generic8_t _var_123;
  generic64_t _var_124;
  generic8_t _var_125;
  generic64_t _var_126;
  generic16_t _var_127;
  generic64_t _var_128;
  generic16_t _var_129;
  generic64_t _var_130;
  generic16_t _var_131;
  generic64_t _var_132;
  generic16_t _var_133;
  generic64_t _var_134;
  generic16_t _var_135;
  generic64_t _var_136;
  generic16_t _var_137;
  generic64_t _var_138;
  generic16_t _var_139;
  generic64_t _var_140;
  generic16_t _var_141;
  generic64_t _var_142;
  generic16_t _var_143;
  generic8_t _var_144;
  generic64_t _var_145;
  generic16_t _var_146;
  generic64_t _var_147;
  generic16_t _var_148;
  generic64_t _var_149;
  generic16_t _var_150;
  generic64_t _var_151;
  generic16_t _var_152;
  generic64_t _var_153;
  generic16_t _var_154;
  generic64_t _var_155;
  generic16_t _var_156;
  generic64_t _var_157;
  generic16_t _var_158;
  generic64_t _var_159;
  generic16_t _var_160;
  generic64_t _var_161;
  generic16_t _var_162;
  generic32_t _var_163;
  generic8_t _var_164;
  generic8_t _var_165;
  generic8_t _var_166;
  generic8_t _var_167;
  generic8_t _var_168;
  generic8_t _var_169;
  generic8_t _var_170;
  generic8_t _var_171;
  generic32_t _var_172;
  generic8_t _var_173;
  generic8_t _var_174;
  generic8_t _var_175;
  generic8_t _var_176;
  generic8_t _var_177;
  generic8_t _var_178;
  generic8_t _var_179;
  generic8_t _var_180;
  generic64_t _var_181;
  generic16_t _var_182;
  generic64_t _var_183;
  generic16_t _var_184;
  generic64_t _var_185;
  generic16_t _var_186;
  generic64_t _var_187;
  generic16_t _var_188;
  generic64_t _var_189;
  generic16_t _var_190;
  generic64_t _var_191;
  generic16_t _var_192;
  generic64_t _var_193;
  generic16_t _var_194;
  generic64_t _var_195;
  generic16_t _var_196;
  generic8_t _var_197;
  generic32_t _var_198;
  generic8_t _var_199;
  generic8_t _var_200;
  generic8_t _var_201;
  generic8_t _var_202;
  generic8_t _var_203;
  generic8_t _var_204;
  generic8_t _var_205;
  generic8_t _var_206;
  generic64_t _var_207;
  generic16_t _var_208;
  generic64_t _var_209;
  generic16_t _var_210;
  generic64_t _var_211;
  generic16_t _var_212;
  generic64_t _var_213;
  generic16_t _var_214;
  generic64_t _var_215;
  generic16_t _var_216;
  generic64_t _var_217;
  generic16_t _var_218;
  generic64_t _var_219;
  generic16_t _var_220;
  generic64_t _var_221;
  generic16_t _var_222;
  generic8_t _var_223;
  generic32_t _var_224;
  generic8_t _var_225;
  generic8_t _var_226;
  generic8_t _var_227;
  generic8_t _var_228;
  generic8_t _var_229;
  generic8_t _var_230;
  generic8_t _var_231;
  generic8_t _var_232;
  generic64_t _var_233;
  generic16_t _var_234;
  generic64_t _var_235;
  generic16_t _var_236;
  generic64_t _var_237;
  generic16_t _var_238;
  generic64_t _var_239;
  generic16_t _var_240;
  generic64_t _var_241;
  generic16_t _var_242;
  generic64_t _var_243;
  generic16_t _var_244;
  generic64_t _var_245;
  generic16_t _var_246;
  generic64_t _var_247;
  generic16_t _var_248;
  generic8_t _var_249;
  generic32_t _var_250;
  generic8_t _var_251;
  generic8_t _var_252;
  generic8_t _var_253;
  generic8_t _var_254;
  generic8_t _var_255;
  generic8_t _var_256;
  generic8_t _var_257;
  generic8_t _var_258;
  generic64_t _var_259;
  generic16_t _var_260;
  generic64_t _var_261;
  generic16_t _var_262;
  generic64_t _var_263;
  generic16_t _var_264;
  generic64_t _var_265;
  generic16_t _var_266;
  generic64_t _var_267;
  generic16_t _var_268;
  generic64_t _var_269;
  generic16_t _var_270;
  generic64_t _var_271;
  generic16_t _var_272;
  generic64_t _var_273;
  generic16_t _var_274;
  generic32_t _var_275;
  generic8_t _var_276;
  generic8_t _var_277;
  generic8_t _var_278;
  generic8_t _var_279;
  generic8_t _var_280;
  generic8_t _var_281;
  generic8_t _var_282;
  generic8_t _var_283;
  generic64_t _var_284;
  generic16_t _var_285;
  generic64_t _var_286;
  generic16_t _var_287;
  generic64_t _var_288;
  generic16_t _var_289;
  generic64_t _var_290;
  generic16_t _var_291;
  generic64_t _var_292;
  generic16_t _var_293;
  generic64_t _var_294;
  generic16_t _var_295;
  generic64_t _var_296;
  generic16_t _var_297;
  generic64_t _var_298;
  generic16_t _var_299;
  generic32_t _var_300;
  generic8_t _var_301;
  generic8_t _var_302;
  generic8_t _var_303;
  generic8_t _var_304;
  generic8_t _var_305;
  generic8_t _var_306;
  generic8_t _var_307;
  generic8_t _var_308;
  generic64_t _var_309;
  generic16_t _var_310;
  generic64_t _var_311;
  generic16_t _var_312;
  generic64_t _var_313;
  generic16_t _var_314;
  generic64_t _var_315;
  generic16_t _var_316;
  generic64_t _var_317;
  generic16_t _var_318;
  generic64_t _var_319;
  generic16_t _var_320;
  generic64_t _var_321;
  generic16_t _var_322;
  generic64_t _var_323;
  generic16_t _var_324;
  generic32_t _var_325;
  generic8_t _var_326;
  generic8_t _var_327;
  generic8_t _var_328;
  generic8_t _var_329;
  generic8_t _var_330;
  generic8_t _var_331;
  generic8_t _var_332;
  generic8_t _var_333;
  generic64_t _var_334;
  generic16_t _var_335;
  generic64_t _var_336;
  generic16_t _var_337;
  generic64_t _var_338;
  generic16_t _var_339;
  generic64_t _var_340;
  generic16_t _var_341;
  generic64_t _var_342;
  generic16_t _var_343;
  generic64_t _var_344;
  generic16_t _var_345;
  generic64_t _var_346;
  generic16_t _var_347;
  generic64_t _var_348;
  generic16_t _var_349;
  generic32_t _var_350;
  generic8_t _var_351;
  generic8_t _var_352;
  generic8_t _var_353;
  generic8_t _var_354;
  generic8_t _var_355;
  generic8_t _var_356;
  generic8_t _var_357;
  generic8_t _var_358;
  generic64_t _var_359;
  generic16_t _var_360;
  generic64_t _var_361;
  generic16_t _var_362;
  generic64_t _var_363;
  generic16_t _var_364;
  generic64_t _var_365;
  generic16_t _var_366;
  generic64_t _var_367;
  generic16_t _var_368;
  generic64_t _var_369;
  generic16_t _var_370;
  generic64_t _var_371;
  generic16_t _var_372;
  generic64_t _var_373;
  generic16_t _var_374;
  generic32_t _var_375;
  generic8_t _var_376;
  generic8_t _var_377;
  generic8_t _var_378;
  generic8_t _var_379;
  generic8_t _var_380;
  generic8_t _var_381;
  generic8_t _var_382;
  generic8_t _var_383;
  generic64_t _var_384;
  generic16_t _var_385;
  generic64_t _var_386;
  generic16_t _var_387;
  generic64_t _var_388;
  generic16_t _var_389;
  generic64_t _var_390;
  generic16_t _var_391;
  generic64_t _var_392;
  generic16_t _var_393;
  generic64_t _var_394;
  generic16_t _var_395;
  generic64_t _var_396;
  generic16_t _var_397;
  generic64_t _var_398;
  generic16_t _var_399;
  generic32_t _var_400;
  generic8_t _var_401;
  generic8_t _var_402;
  generic8_t _var_403;
  generic8_t _var_404;
  generic8_t _var_405;
  generic8_t _var_406;
  generic8_t _var_407;
  generic8_t _var_408;
  generic64_t _var_409;
  generic16_t _var_410;
  generic64_t _var_411;
  generic16_t _var_412;
  generic64_t _var_413;
  generic16_t _var_414;
  generic64_t _var_415;
  generic16_t _var_416;
  generic64_t _var_417;
  generic16_t _var_418;
  generic64_t _var_419;
  generic16_t _var_420;
  generic64_t _var_421;
  generic16_t _var_422;
  generic64_t _var_423;
  generic16_t _var_424;
  generic32_t _var_425;
  generic8_t _var_426;
  generic8_t _var_427;
  generic8_t _var_428;
  generic8_t _var_429;
  generic8_t _var_430;
  generic8_t _var_431;
  generic8_t _var_432;
  generic8_t _var_433;
  generic64_t _var_434;
  generic16_t _var_435;
  generic64_t _var_436;
  generic16_t _var_437;
  generic64_t _var_438;
  generic16_t _var_439;
  generic64_t _var_440;
  generic16_t _var_441;
  generic64_t _var_442;
  generic16_t _var_443;
  generic64_t _var_444;
  generic16_t _var_445;
  generic64_t _var_446;
  generic16_t _var_447;
  generic64_t _var_448;
  generic16_t _var_449;
  generic32_t _var_450;
  generic8_t _var_451;
  generic8_t _var_452;
  generic8_t _var_453;
  generic8_t _var_454;
  generic8_t _var_455;
  generic8_t _var_456;
  generic8_t _var_457;
  generic8_t _var_458;
  generic64_t _var_459;
  generic16_t _var_460;
  generic64_t _var_461;
  generic16_t _var_462;
  generic64_t _var_463;
  generic16_t _var_464;
  generic64_t _var_465;
  generic16_t _var_466;
  generic64_t _var_467;
  generic16_t _var_468;
  generic64_t _var_469;
  generic16_t _var_470;
  generic64_t _var_471;
  generic16_t _var_472;
  generic64_t _var_473;
  generic16_t _var_474;
  generic64_t _var_475;
  generic8_t _var_476;
  generic64_t _var_477;
  generic16_t _var_478;
  generic64_t _var_479;
  generic8_t _var_480;
  generic64_t _var_481;
  generic16_t _var_482;
  generic32_t _var_483;
  generic8_t _var_484;
  generic8_t _var_485;
  generic8_t _var_486;
  generic8_t _var_487;
  generic8_t _var_488;
  generic8_t _var_489;
  generic8_t _var_490;
  generic8_t _var_491;
  generic64_t _var_492;
  generic16_t _var_493;
  generic64_t _var_494;
  generic16_t _var_495;
  generic64_t _var_496;
  generic16_t _var_497;
  generic64_t _var_498;
  generic16_t _var_499;
  generic64_t _var_500;
  generic16_t _var_501;
  generic64_t _var_502;
  generic16_t _var_503;
  generic64_t _var_504;
  generic16_t _var_505;
  generic64_t _var_506;
  generic16_t _var_507;
  generic8_t _var_508;
  generic64_t _var_509;
  generic16_t _var_510;
  generic64_t _var_511;
  generic16_t _var_512;
  generic64_t _var_513;
  generic16_t _var_514;
  generic64_t _var_515;
  generic16_t _var_516;
  generic64_t _var_517;
  generic16_t _var_518;
  generic64_t _var_519;
  generic16_t _var_520;
  generic64_t _var_521;
  generic16_t _var_522;
  generic64_t _var_523;
  generic16_t _var_524;
  generic8_t _var_525;
  generic64_t _var_526;
  generic16_t _var_527;
  generic64_t _var_528;
  generic16_t _var_529;
  generic64_t _var_530;
  generic16_t _var_531;
  generic64_t _var_532;
  generic16_t _var_533;
  generic64_t _var_534;
  generic16_t _var_535;
  generic64_t _var_536;
  generic16_t _var_537;
  generic64_t _var_538;
  generic16_t _var_539;
  generic64_t _var_540;
  generic16_t _var_541;
  generic32_t _var_542;
  generic8_t _var_543;
  generic8_t _var_544;
  generic8_t _var_545;
  generic8_t _var_546;
  generic8_t _var_547;
  generic8_t _var_548;
  generic8_t _var_549;
  generic8_t _var_550;
  generic64_t _var_551;
  generic16_t _var_552;
  generic64_t _var_553;
  generic16_t _var_554;
  generic64_t _var_555;
  generic16_t _var_556;
  generic64_t _var_557;
  generic16_t _var_558;
  generic64_t _var_559;
  generic16_t _var_560;
  generic64_t _var_561;
  generic16_t _var_562;
  generic64_t _var_563;
  generic16_t _var_564;
  generic64_t _var_565;
  generic16_t _var_566;
  generic8_t _var_567;
  generic64_t _var_568;
  generic16_t _var_569;
  generic64_t _var_570;
  generic16_t _var_571;
  generic64_t _var_572;
  generic16_t _var_573;
  generic64_t _var_574;
  generic16_t _var_575;
  generic64_t _var_576;
  generic16_t _var_577;
  generic64_t _var_578;
  generic16_t _var_579;
  generic64_t _var_580;
  generic16_t _var_581;
  generic64_t _var_582;
  generic16_t _var_583;
  generic8_t _var_584;
  generic64_t _var_585;
  generic16_t _var_586;
  generic64_t _var_587;
  generic16_t _var_588;
  generic64_t _var_589;
  generic16_t _var_590;
  generic64_t _var_591;
  generic16_t _var_592;
  generic64_t _var_593;
  generic16_t _var_594;
  generic64_t _var_595;
  generic16_t _var_596;
  generic64_t _var_597;
  generic16_t _var_598;
  generic64_t _var_599;
  generic16_t _var_600;
  generic64_t _var_601;
  generic16_t _var_602;
  generic64_t _var_603;
  generic16_t _var_604;
  generic64_t _var_605;
  generic16_t _var_606;
  generic64_t _var_607;
  generic16_t _var_608;
  generic64_t _var_609;
  generic16_t _var_610;
  generic64_t _var_611;
  generic16_t _var_612;
  generic64_t _var_613;
  generic16_t _var_614;
  generic64_t _var_615;
  generic16_t _var_616;
  generic64_t _var_617;
  generic16_t _var_618;
  generic64_t _var_619;
  generic16_t _var_620;
  generic64_t _var_621;
  generic16_t _var_622;
  generic64_t _var_623;
  generic16_t _var_624;
  generic64_t _var_625;
  generic16_t _var_626;
  generic64_t _var_627;
  generic16_t _var_628;
  generic64_t _var_629;
  generic16_t _var_630;
  generic64_t _var_631;
  generic16_t _var_632;
  generic64_t _var_633;
  generic16_t _var_634;
  generic8_t _var_635;
  generic64_t _var_636;
  generic16_t _var_637;
  generic64_t _var_638;
  generic16_t _var_639;
  generic64_t _var_640;
  generic16_t _var_641;
  generic64_t _var_642;
  generic16_t _var_643;
  generic64_t _var_644;
  generic16_t _var_645;
  generic64_t _var_646;
  generic16_t _var_647;
  generic64_t _var_648;
  generic16_t _var_649;
  generic64_t _var_650;
  generic16_t _var_651;
  generic64_t _var_652;
  generic16_t _var_653;
  generic8_t _var_654;
  generic64_t _var_655;
  generic16_t _var_656;
  generic16_t _var_657;
  generic8_t _var_658;
  generic8_t _var_659;
  generic32_t _var_660;
  generic8_t _var_661;
  generic8_t _var_662;
  generic8_t _var_663;
  generic8_t _var_664;
  generic8_t _var_665;
  generic8_t _var_666;
  generic8_t _var_667;
  generic8_t _var_668;
  generic8_t _var_669;
  generic16_t _var_670;
  generic8_t _var_671;
  generic8_t _var_672;
  generic64_t _var_673;
  generic16_t _var_674;
  generic64_t _var_675;
  generic16_t _var_676;
  generic64_t _var_677;
  generic16_t _var_678;
  generic64_t _var_679;
  generic16_t _var_680;
  generic64_t _var_681;
  generic16_t _var_682;
  generic64_t _var_683;
  generic16_t _var_684;
  generic64_t _var_685;
  generic16_t _var_686;
  generic64_t _var_687;
  generic16_t _var_688;
  generic32_t _var_689;
  generic8_t _var_690;
  generic8_t _var_691;
  generic8_t _var_692;
  generic8_t _var_693;
  generic8_t _var_694;
  generic8_t _var_695;
  generic8_t _var_696;
  generic8_t _var_697;
  generic64_t _var_698;
  generic8_t _var_699;
  generic64_t _var_700;
  generic16_t _var_701;
  generic64_t _var_702;
  generic16_t _var_703;
  generic64_t _var_704;
  generic16_t _var_705;
  generic64_t _var_706;
  generic16_t _var_707;
  generic64_t _var_708;
  generic16_t _var_709;
  generic64_t _var_710;
  generic16_t _var_711;
  generic64_t _var_712;
  generic16_t _var_713;
  generic64_t _var_714;
  generic16_t _var_715;
  generic64_t _var_716;
  generic16_t _var_717;
  generic8_t _var_718;
  generic8_t _var_719;
  generic64_t _var_720;
  generic16_t _var_721;
  generic32_t _var_722;
  generic8_t _var_723;
  generic8_t _var_724;
  generic8_t _var_725;
  generic8_t _var_726;
  generic8_t _var_727;
  generic8_t _var_728;
  generic8_t _var_729;
  generic8_t _var_730;
  generic64_t _var_731;
  generic16_t _var_732;
  generic64_t _var_733;
  generic16_t _var_734;
  generic64_t _var_735;
  generic16_t _var_736;
  generic64_t _var_737;
  generic16_t _var_738;
  generic64_t _var_739;
  generic16_t _var_740;
  generic64_t _var_741;
  generic16_t _var_742;
  generic64_t _var_743;
  generic16_t _var_744;
  generic64_t _var_745;
  generic16_t _var_746;
  generic8_t _var_747;
  generic32_t _var_748;
  generic8_t _var_749;
  generic8_t _var_750;
  generic8_t _var_751;
  generic8_t _var_752;
  generic8_t _var_753;
  generic8_t _var_754;
  generic8_t _var_755;
  generic8_t _var_756;
  generic64_t _var_757;
  generic16_t _var_758;
  generic64_t _var_759;
  generic16_t _var_760;
  generic64_t _var_761;
  generic16_t _var_762;
  generic64_t _var_763;
  generic16_t _var_764;
  generic64_t _var_765;
  generic16_t _var_766;
  generic64_t _var_767;
  generic16_t _var_768;
  generic64_t _var_769;
  generic16_t _var_770;
  generic64_t _var_771;
  generic16_t _var_772;
  generic16_t _var_773;
  generic8_t _var_774;
  generic8_t _var_775;
  generic32_t _var_776;
  generic8_t _var_777;
  generic8_t _var_778;
  generic8_t _var_779;
  generic8_t _var_780;
  generic8_t _var_781;
  generic8_t _var_782;
  generic8_t _var_783;
  generic8_t _var_784;
  generic8_t _var_785;
  generic16_t _var_786;
  generic8_t _var_787;
  generic8_t _var_788;
  generic64_t _var_789;
  generic16_t _var_790;
  generic64_t _var_791;
  generic16_t _var_792;
  generic64_t _var_793;
  generic16_t _var_794;
  generic64_t _var_795;
  generic16_t _var_796;
  generic64_t _var_797;
  generic16_t _var_798;
  generic64_t _var_799;
  generic16_t _var_800;
  generic64_t _var_801;
  generic16_t _var_802;
  generic64_t _var_803;
  generic16_t _var_804;
  generic32_t _var_805;
  generic8_t _var_806;
  generic8_t _var_807;
  generic8_t _var_808;
  generic8_t _var_809;
  generic8_t _var_810;
  generic8_t _var_811;
  generic8_t _var_812;
  generic8_t _var_813;
  generic64_t _var_814;
  generic16_t _var_815;
  generic64_t _var_816;
  generic16_t _var_817;
  generic64_t _var_818;
  generic16_t _var_819;
  generic64_t _var_820;
  generic16_t _var_821;
  generic64_t _var_822;
  generic16_t _var_823;
  generic64_t _var_824;
  generic16_t _var_825;
  generic64_t _var_826;
  generic16_t _var_827;
  generic64_t _var_828;
  generic16_t _var_829;
  generic64_t _var_830;
  generic16_t _var_831;
  generic64_t _var_832;
  generic16_t _var_833;
  generic64_t _var_834;
  generic16_t _var_835;
  generic64_t _var_836;
  generic16_t _var_837;
  generic64_t _var_838;
  generic16_t _var_839;
  generic64_t _var_840;
  generic16_t _var_841;
  generic64_t _var_842;
  generic16_t _var_843;
  generic64_t _var_844;
  generic16_t _var_845;
  generic8_t _var_846;
  generic64_t _var_847;
  generic16_t _var_848;
  generic32_t _var_849;
  generic8_t _var_850;
  generic8_t _var_851;
  generic8_t _var_852;
  generic8_t _var_853;
  generic8_t _var_854;
  generic8_t _var_855;
  generic8_t _var_856;
  generic8_t _var_857;
  generic64_t _var_858;
  generic16_t _var_859;
  generic64_t _var_860;
  generic16_t _var_861;
  generic64_t _var_862;
  generic16_t _var_863;
  generic64_t _var_864;
  generic16_t _var_865;
  generic64_t _var_866;
  generic16_t _var_867;
  generic64_t _var_868;
  generic16_t _var_869;
  generic64_t _var_870;
  generic16_t _var_871;
  generic64_t _var_872;
  generic16_t _var_873;
  generic32_t _var_874;
  generic8_t _var_875;
  generic8_t _var_876;
  generic8_t _var_877;
  generic8_t _var_878;
  generic8_t _var_879;
  generic8_t _var_880;
  generic8_t _var_881;
  generic8_t _var_882;
  generic64_t _var_883;
  generic16_t _var_884;
  generic64_t _var_885;
  generic16_t _var_886;
  generic64_t _var_887;
  generic16_t _var_888;
  generic64_t _var_889;
  generic16_t _var_890;
  generic64_t _var_891;
  generic16_t _var_892;
  generic64_t _var_893;
  generic16_t _var_894;
  generic64_t _var_895;
  generic16_t _var_896;
  generic64_t _var_897;
  generic16_t _var_898;
  generic64_t _var_899;
  generic16_t _var_900;
  generic64_t _var_901;
  generic16_t _var_902;
  generic64_t _var_903;
  generic16_t _var_904;
  generic64_t _var_905;
  generic16_t _var_906;
  generic64_t _var_907;
  generic16_t _var_908;
  generic64_t _var_909;
  generic16_t _var_910;
  generic64_t _var_911;
  generic16_t _var_912;
  generic64_t _var_913;
  generic16_t _var_914;
  generic32_t _var_915;
  generic8_t _var_916;
  generic8_t _var_917;
  generic8_t _var_918;
  generic8_t _var_919;
  generic8_t _var_920;
  generic8_t _var_921;
  generic8_t _var_922;
  generic8_t _var_923;
  generic32_t _var_924;
  generic8_t _var_925;
  generic8_t _var_926;
  generic8_t _var_927;
  generic8_t _var_928;
  generic8_t _var_929;
  generic8_t _var_930;
  generic8_t _var_931;
  generic8_t _var_932;
  generic64_t _var_933;
  generic16_t _var_934;
  generic64_t _var_935;
  generic16_t _var_936;
  generic64_t _var_937;
  generic16_t _var_938;
  generic64_t _var_939;
  generic16_t _var_940;
  generic64_t _var_941;
  generic16_t _var_942;
  generic64_t _var_943;
  generic16_t _var_944;
  generic64_t _var_945;
  generic16_t _var_946;
  generic64_t _var_947;
  generic16_t _var_948;
  generic32_t _var_949;
  generic8_t _var_950;
  generic8_t _var_951;
  generic8_t _var_952;
  generic8_t _var_953;
  generic8_t _var_954;
  generic8_t _var_955;
  generic8_t _var_956;
  generic8_t _var_957;
  generic64_t _var_958;
  generic16_t _var_959;
  generic64_t _var_960;
  generic16_t _var_961;
  generic64_t _var_962;
  generic16_t _var_963;
  generic64_t _var_964;
  generic16_t _var_965;
  generic64_t _var_966;
  generic16_t _var_967;
  generic64_t _var_968;
  generic16_t _var_969;
  generic64_t _var_970;
  generic16_t _var_971;
  generic64_t _var_972;
  generic16_t _var_973;
  generic8_t _var_974;
  generic8_t _var_975;
  generic64_t _var_976;
  generic16_t _var_977;
  generic32_t _var_978;
  generic8_t _var_979;
  generic8_t _var_980;
  generic8_t _var_981;
  generic8_t _var_982;
  generic8_t _var_983;
  generic8_t _var_984;
  generic8_t _var_985;
  generic8_t _var_986;
  generic64_t _var_987;
  generic16_t _var_988;
  generic64_t _var_989;
  generic16_t _var_990;
  generic64_t _var_991;
  generic16_t _var_992;
  generic64_t _var_993;
  generic16_t _var_994;
  generic64_t _var_995;
  generic16_t _var_996;
  generic64_t _var_997;
  generic16_t _var_998;
  generic64_t _var_999;
  generic16_t _var_1000;
  generic64_t _var_1001;
  generic16_t _var_1002;
  generic8_t _var_1003;
  generic32_t _var_1004;
  generic8_t _var_1005;
  generic8_t _var_1006;
  generic8_t _var_1007;
  generic8_t _var_1008;
  generic8_t _var_1009;
  generic8_t _var_1010;
  generic8_t _var_1011;
  generic8_t _var_1012;
  generic64_t _var_1013;
  generic16_t _var_1014;
  generic64_t _var_1015;
  generic16_t _var_1016;
  generic64_t _var_1017;
  generic16_t _var_1018;
  generic64_t _var_1019;
  generic16_t _var_1020;
  generic64_t _var_1021;
  generic16_t _var_1022;
  generic64_t _var_1023;
  generic16_t _var_1024;
  generic64_t _var_1025;
  generic16_t _var_1026;
  generic64_t _var_1027;
  generic16_t _var_1028;
  generic64_t _var_1029;
  generic8_t _var_1030;
  generic64_t _var_1031;
  generic16_t _var_1032;
  generic32_t _var_1033;
  generic8_t _var_1034;
  generic8_t _var_1035;
  generic8_t _var_1036;
  generic8_t _var_1037;
  generic8_t _var_1038;
  generic8_t _var_1039;
  generic8_t _var_1040;
  generic8_t _var_1041;
  generic64_t _var_1042;
  generic16_t _var_1043;
  generic64_t _var_1044;
  generic16_t _var_1045;
  generic64_t _var_1046;
  generic16_t _var_1047;
  generic64_t _var_1048;
  generic16_t _var_1049;
  generic64_t _var_1050;
  generic16_t _var_1051;
  generic64_t _var_1052;
  generic16_t _var_1053;
  generic64_t _var_1054;
  generic16_t _var_1055;
  generic64_t _var_1056;
  generic16_t _var_1057;
  generic8_t _var_1058;
  generic32_t _var_1059;
  generic8_t _var_1060;
  generic8_t _var_1061;
  generic8_t _var_1062;
  generic8_t _var_1063;
  generic8_t _var_1064;
  generic8_t _var_1065;
  generic8_t _var_1066;
  generic8_t _var_1067;
  generic64_t _var_1068;
  generic16_t _var_1069;
  generic64_t _var_1070;
  generic16_t _var_1071;
  generic64_t _var_1072;
  generic16_t _var_1073;
  generic64_t _var_1074;
  generic16_t _var_1075;
  generic64_t _var_1076;
  generic16_t _var_1077;
  generic64_t _var_1078;
  generic16_t _var_1079;
  generic64_t _var_1080;
  generic16_t _var_1081;
  generic64_t _var_1082;
  generic16_t _var_1083;
  generic64_t _var_1084;
  generic8_t _var_1085;
  generic64_t _var_1086;
  generic16_t _var_1087;
  generic64_t _var_1088;
  generic16_t _var_1089;
  generic64_t _var_1090;
  generic16_t _var_1091;
  generic64_t _var_1092;
  generic16_t _var_1093;
  generic64_t _var_1094;
  generic16_t _var_1095;
  generic64_t _var_1096;
  generic16_t _var_1097;
  generic64_t _var_1098;
  generic16_t _var_1099;
  generic64_t _var_1100;
  generic16_t _var_1101;
  generic64_t _var_1102;
  generic16_t _var_1103;
  generic64_t _var_1104;
  generic16_t _var_1105;
  generic64_t _var_1106;
  generic16_t _var_1107;
  generic64_t _var_1108;
  generic16_t _var_1109;
  generic64_t _var_1110;
  generic16_t _var_1111;
  generic64_t _var_1112;
  generic16_t _var_1113;
  generic64_t _var_1114;
  generic16_t _var_1115;
  generic64_t _var_1116;
  generic16_t _var_1117;
  generic64_t _var_1118;
  generic16_t _var_1119;
  generic32_t _var_1120;
  generic8_t _var_1121;
  generic8_t _var_1122;
  generic8_t _var_1123;
  generic8_t _var_1124;
  generic8_t _var_1125;
  generic8_t _var_1126;
  generic8_t _var_1127;
  generic8_t _var_1128;
  generic64_t _var_1129;
  generic16_t _var_1130;
  generic64_t _var_1131;
  generic16_t _var_1132;
  generic64_t _var_1133;
  generic16_t _var_1134;
  generic64_t _var_1135;
  generic16_t _var_1136;
  generic64_t _var_1137;
  generic16_t _var_1138;
  generic64_t _var_1139;
  generic16_t _var_1140;
  generic64_t _var_1141;
  generic16_t _var_1142;
  generic64_t _var_1143;
  generic16_t _var_1144;
  generic8_t _var_1145;
  generic64_t _var_1146;
  generic16_t _var_1147;
  generic32_t _var_1148;
  generic8_t _var_1149;
  generic8_t _var_1150;
  generic8_t _var_1151;
  generic8_t _var_1152;
  generic8_t _var_1153;
  generic8_t _var_1154;
  generic8_t _var_1155;
  generic8_t _var_1156;
  generic64_t _var_1157;
  generic8_t _var_1158;
  generic64_t _var_1159;
  generic16_t _var_1160;
  generic32_t _var_1161;
  generic8_t _var_1162;
  generic8_t _var_1163;
  generic8_t _var_1164;
  generic8_t _var_1165;
  generic8_t _var_1166;
  generic8_t _var_1167;
  generic8_t _var_1168;
  generic8_t _var_1169;
  generic32_t _var_1170;
  generic8_t _var_1171;
  generic8_t _var_1172;
  generic8_t _var_1173;
  generic8_t _var_1174;
  generic8_t _var_1175;
  generic8_t _var_1176;
  generic8_t _var_1177;
  generic8_t _var_1178;
  generic64_t _var_1179;
  generic16_t _var_1180;
  generic64_t _var_1181;
  generic16_t _var_1182;
  generic64_t _var_1183;
  generic16_t _var_1184;
  generic64_t _var_1185;
  generic16_t _var_1186;
  generic64_t _var_1187;
  generic16_t _var_1188;
  generic64_t _var_1189;
  generic16_t _var_1190;
  generic64_t _var_1191;
  generic16_t _var_1192;
  generic64_t _var_1193;
  generic16_t _var_1194;
  generic32_t _var_1195;
  generic8_t _var_1196;
  generic8_t _var_1197;
  generic8_t _var_1198;
  generic8_t _var_1199;
  generic8_t _var_1200;
  generic8_t _var_1201;
  generic8_t _var_1202;
  generic8_t _var_1203;
  generic32_t _var_1204;
  generic8_t _var_1205;
  generic8_t _var_1206;
  generic8_t _var_1207;
  generic8_t _var_1208;
  generic8_t _var_1209;
  generic8_t _var_1210;
  generic8_t _var_1211;
  generic8_t _var_1212;
  generic64_t _var_1213;
  generic16_t _var_1214;
  generic64_t _var_1215;
  generic16_t _var_1216;
  generic64_t _var_1217;
  generic16_t _var_1218;
  generic64_t _var_1219;
  generic16_t _var_1220;
  generic64_t _var_1221;
  generic16_t _var_1222;
  generic64_t _var_1223;
  generic16_t _var_1224;
  generic64_t _var_1225;
  generic16_t _var_1226;
  generic64_t _var_1227;
  generic16_t _var_1228;
  generic32_t _var_1229;
  generic8_t _var_1230;
  generic8_t _var_1231;
  generic8_t _var_1232;
  generic8_t _var_1233;
  generic8_t _var_1234;
  generic8_t _var_1235;
  generic8_t _var_1236;
  generic8_t _var_1237;
  generic64_t _var_1238;
  generic16_t _var_1239;
  generic64_t _var_1240;
  generic16_t _var_1241;
  generic64_t _var_1242;
  generic16_t _var_1243;
  generic64_t _var_1244;
  generic16_t _var_1245;
  generic64_t _var_1246;
  generic16_t _var_1247;
  generic64_t _var_1248;
  generic16_t _var_1249;
  generic64_t _var_1250;
  generic16_t _var_1251;
  generic64_t _var_1252;
  generic16_t _var_1253;
  generic32_t _var_1254;
  generic8_t _var_1255;
  generic8_t _var_1256;
  generic8_t _var_1257;
  generic8_t _var_1258;
  generic8_t _var_1259;
  generic8_t _var_1260;
  generic8_t _var_1261;
  generic8_t _var_1262;
  generic64_t _var_1263;
  generic16_t _var_1264;
  generic64_t _var_1265;
  generic16_t _var_1266;
  generic64_t _var_1267;
  generic16_t _var_1268;
  generic64_t _var_1269;
  generic16_t _var_1270;
  generic64_t _var_1271;
  generic16_t _var_1272;
  generic64_t _var_1273;
  generic16_t _var_1274;
  generic64_t _var_1275;
  generic16_t _var_1276;
  generic64_t _var_1277;
  generic16_t _var_1278;
  generic32_t _var_1279;
  generic8_t _var_1280;
  generic8_t _var_1281;
  generic8_t _var_1282;
  generic8_t _var_1283;
  generic8_t _var_1284;
  generic8_t _var_1285;
  generic8_t _var_1286;
  generic8_t _var_1287;
  generic32_t _var_1288;
  generic8_t _var_1289;
  generic8_t _var_1290;
  generic8_t _var_1291;
  generic8_t _var_1292;
  generic8_t _var_1293;
  generic8_t _var_1294;
  generic8_t _var_1295;
  generic8_t _var_1296;
  generic64_t _var_1297;
  generic16_t _var_1298;
  generic64_t _var_1299;
  generic16_t _var_1300;
  generic64_t _var_1301;
  generic16_t _var_1302;
  generic64_t _var_1303;
  generic16_t _var_1304;
  generic64_t _var_1305;
  generic16_t _var_1306;
  generic64_t _var_1307;
  generic16_t _var_1308;
  generic64_t _var_1309;
  generic16_t _var_1310;
  generic64_t _var_1311;
  generic16_t _var_1312;
  generic32_t _var_1313;
  generic8_t _var_1314;
  generic8_t _var_1315;
  generic8_t _var_1316;
  generic8_t _var_1317;
  generic8_t _var_1318;
  generic8_t _var_1319;
  generic8_t _var_1320;
  generic8_t _var_1321;
  generic32_t _var_1322;
  generic8_t _var_1323;
  generic8_t _var_1324;
  generic8_t _var_1325;
  generic8_t _var_1326;
  generic8_t _var_1327;
  generic8_t _var_1328;
  generic8_t _var_1329;
  generic8_t _var_1330;
  generic64_t _var_1331;
  generic16_t _var_1332;
  generic64_t _var_1333;
  generic16_t _var_1334;
  generic64_t _var_1335;
  generic16_t _var_1336;
  generic64_t _var_1337;
  generic16_t _var_1338;
  generic64_t _var_1339;
  generic16_t _var_1340;
  generic64_t _var_1341;
  generic16_t _var_1342;
  generic64_t _var_1343;
  generic16_t _var_1344;
  generic64_t _var_1345;
  generic16_t _var_1346;
  uint8_t *_var_1347;
  generic64_t _var_1348;
  generic16_t _var_1349;
  generic64_t _var_1350;
  generic16_t _var_1351;
  generic64_t _var_1352;
  generic16_t _var_1353;
  generic64_t _var_1354;
  generic16_t _var_1355;
  generic64_t _var_1356;
  generic16_t _var_1357;
  generic64_t _var_1358;
  generic16_t _var_1359;
  generic64_t _var_1360;
  generic16_t _var_1361;
  generic64_t _var_1362;
  generic16_t _var_1363;
  generic32_t _var_1364;
  generic32_t _var_1365;
  generic32_t _var_1366;
  _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 7536, 0, &_var_1322, &_var_1323, &_var_1324, &_var_1325, &_var_1326, &_var_1327, &_var_1328, &_var_1329, &_var_1330, &_var_1331, &_var_1332, &_var_1333, &_var_1334, &_var_1335, &_var_1336, &_var_1337, &_var_1338, &_var_1339, &_var_1340, &_var_1341, &_var_1342, &_var_1343, &_var_1344, &_var_1345, &_var_1346);
  _stack.offset_16 = w;
  *((int32_t *) &_stack.offset_20) = fl;
  _stack.offset_24.member_1 = t;
  _helper_fpush_wrapper(NULL, _var_1322, &_var_1313, &_var_1314, &_var_1315, &_var_1316, &_var_1317, &_var_1318, &_var_1319, &_var_1320, &_var_1321);
  _helper_fmov_ST0_STN_wrapper(NULL, 1, _var_1313, _var_1331, _var_1332, _var_1333, _var_1334, _var_1335, _var_1336, _var_1337, _var_1338, _var_1339, _var_1340, _var_1341, _var_1342, _var_1343, _var_1344, _var_1345, _var_1346, &_var_1297, &_var_1298, &_var_1299, &_var_1300, &_var_1301, &_var_1302, &_var_1303, &_var_1304, &_var_1305, &_var_1306, &_var_1307, &_var_1308, &_var_1309, &_var_1310, &_var_1311, &_var_1312);
  _helper_fstt_ST0_wrapper(NULL, &_stack, _var_1313, _var_1297, _var_1298, _var_1299, _var_1300, _var_1301, _var_1302, _var_1303, _var_1304, _var_1305, _var_1306, _var_1307, _var_1308, _var_1309, _var_1310, _var_1311, _var_1312);
  _helper_fpop_wrapper(NULL, _var_1313, &_var_1288, &_var_1289, &_var_1290, &_var_1291, &_var_1292, &_var_1293, &_var_1294, &_var_1295, &_var_1296);
  _stack.offset_88 = 0;
  _helper_fstt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 7536, _var_1288, _var_1297, _var_1298, _var_1299, _var_1300, _var_1301, _var_1302, _var_1303, _var_1304, _var_1305, _var_1306, _var_1307, _var_1308, _var_1309, _var_1310, _var_1311, _var_1312);
  _helper_fpop_wrapper(NULL, _var_1288, &_var_1279, &_var_1280, &_var_1281, &_var_1282, &_var_1283, &_var_1284, &_var_1285, &_var_1286, &_var_1287);
  _var_0 = unreserved___signbitl((float128_t) ((number128_t) y & ((uint128_t) 0xFFFFFFFFFFFFFFFF)));
  _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 7536, _var_1279, &_var_1254, &_var_1255, &_var_1256, &_var_1257, &_var_1258, &_var_1259, &_var_1260, &_var_1261, &_var_1262, &_var_1263, &_var_1264, &_var_1265, &_var_1266, &_var_1267, &_var_1268, &_var_1269, &_var_1270, &_var_1271, &_var_1272, &_var_1273, &_var_1274, &_var_1275, &_var_1276, &_var_1277, &_var_1278);
  if (!_var_0) {
    generic64_t _var_1367;
    _var_1367 = _lshift(0, 4294967272);
    if (!(*((generic32_t *) &_stack.offset_20) & 0x800)) {
      uint8_t *_var_1368;
      _var_1366 = *((generic32_t *) &_stack.offset_20) & 0x1;
      _stack.offset_48.member_0 = _var_1366;
      _var_1368 = !_var_1366 ? (generic64_t) "0X+0X 0X-0x+0x 0x" : (generic64_t) " 0X-0x+0x 0x";
      _var_1347 = _var_1368;
      _var_1348 = _var_1263;
      _var_1349 = _var_1264;
      _var_1350 = _var_1265;
      _var_1351 = _var_1266;
      _var_1352 = _var_1267;
      _var_1353 = _var_1268;
      _var_1354 = _var_1269;
      _var_1355 = _var_1270;
      _var_1356 = _var_1271;
      _var_1357 = _var_1272;
      _var_1358 = _var_1273;
      _var_1359 = _var_1274;
      _var_1360 = _var_1275;
      _var_1361 = _var_1276;
      _var_1362 = _var_1277;
      _var_1363 = _var_1278;
      _var_1364 = ((number32_t) _var_1367 & 0x80) | (((uint32_t) *((generic32_t *) &_stack.offset_20) >> 11) & 0x1) | 0x44;
      _var_1365 = 24;
    } else {
      _stack.offset_48.member_0 = 1;
      _var_1347 = "+0X 0X-0x+0x 0x";
      _var_1348 = _var_1263;
      _var_1349 = _var_1264;
      _var_1350 = _var_1265;
      _var_1351 = _var_1266;
      _var_1352 = _var_1267;
      _var_1353 = _var_1268;
      _var_1354 = _var_1269;
      _var_1355 = _var_1270;
      _var_1356 = _var_1271;
      _var_1357 = _var_1272;
      _var_1358 = _var_1273;
      _var_1359 = _var_1274;
      _var_1360 = _var_1275;
      _var_1361 = _var_1276;
      _var_1362 = _var_1277;
      _var_1363 = _var_1278;
      _var_1364 = ((number32_t) _var_1367 & 0x80) | (((uint32_t) *((generic32_t *) &_stack.offset_20) >> 11) & 0x1) | 0x44;
      _var_1365 = 1;
      _var_1366 = _var_0;
    }
  } else {
    _stack.offset_48.member_0 = 1;
    _helper_fchs_ST0_wrapper(NULL, _var_1254, _var_1263, _var_1264, _var_1265, _var_1266, _var_1267, _var_1268, _var_1269, _var_1270, _var_1271, _var_1272, _var_1273, _var_1274, _var_1275, _var_1276, _var_1277, _var_1278, &_var_1238, &_var_1239, &_var_1240, &_var_1241, &_var_1242, &_var_1243, &_var_1244, &_var_1245, &_var_1246, &_var_1247, &_var_1248, &_var_1249, &_var_1250, &_var_1251, &_var_1252, &_var_1253);
    _var_1348 = _var_1238;
    _var_1349 = _var_1239;
    _var_1350 = _var_1240;
    _var_1351 = _var_1241;
    _var_1352 = _var_1242;
    _var_1353 = _var_1243;
    _var_1354 = _var_1244;
    _var_1355 = _var_1245;
    _var_1356 = _var_1246;
    _var_1357 = _var_1247;
    _var_1358 = _var_1248;
    _var_1359 = _var_1249;
    _var_1360 = _var_1250;
    _var_1361 = _var_1251;
    _var_1362 = _var_1252;
    _var_1363 = _var_1253;
    _var_1347 = "-0X+0X 0X-0x+0x 0x";
    _var_1364 = 7480;
    _var_1365 = 24;
    _var_1366 = _var_0;
  }
  int32_t _var_1369;
  generic32_t _var_1370;
  _stack.offset_8 = (uint64_t) w;
  _stack.offset_0 = (uint64_t) w;
  _helper_fpush_wrapper(NULL, _var_1254, &_var_1229, &_var_1230, &_var_1231, &_var_1232, &_var_1233, &_var_1234, &_var_1235, &_var_1236, &_var_1237);
  _helper_fmov_ST0_STN_wrapper(NULL, 1, _var_1229, _var_1348, _var_1349, _var_1350, _var_1351, _var_1352, _var_1353, _var_1354, _var_1355, _var_1356, _var_1357, _var_1358, _var_1359, _var_1360, _var_1361, _var_1362, _var_1363, &_var_1213, &_var_1214, &_var_1215, &_var_1216, &_var_1217, &_var_1218, &_var_1219, &_var_1220, &_var_1221, &_var_1222, &_var_1223, &_var_1224, &_var_1225, &_var_1226, &_var_1227, &_var_1228);
  _helper_fstt_ST0_wrapper(NULL, &_stack, _var_1229, _var_1213, _var_1214, _var_1215, _var_1216, _var_1217, _var_1218, _var_1219, _var_1220, _var_1221, _var_1222, _var_1223, _var_1224, _var_1225, _var_1226, _var_1227, _var_1228);
  _helper_fpop_wrapper(NULL, _var_1229, &_var_1204, &_var_1205, &_var_1206, &_var_1207, &_var_1208, &_var_1209, &_var_1210, &_var_1211, &_var_1212);
  _helper_fstt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 7536, _var_1204, _var_1213, _var_1214, _var_1215, _var_1216, _var_1217, _var_1218, _var_1219, _var_1220, _var_1221, _var_1222, _var_1223, _var_1224, _var_1225, _var_1226, _var_1227, _var_1228);
  _helper_fpop_wrapper(NULL, _var_1204, &_var_1195, &_var_1196, &_var_1197, &_var_1198, &_var_1199, &_var_1200, &_var_1201, &_var_1202, &_var_1203);
  _var_1369 = unreserved___fpclassifyl((float128_t) ((number128_t) y & ((uint128_t) 0xFFFFFFFFFFFFFFFF)));
  _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 7536, _var_1195, &_var_1170, &_var_1171, &_var_1172, &_var_1173, &_var_1174, &_var_1175, &_var_1176, &_var_1177, &_var_1178, &_var_1179, &_var_1180, &_var_1181, &_var_1182, &_var_1183, &_var_1184, &_var_1185, &_var_1186, &_var_1187, &_var_1188, &_var_1189, &_var_1190, &_var_1191, &_var_1192, &_var_1193, &_var_1194);
  _var_1370 = 0;
  switch ((number32_t) _var_1365) {
    case 9:
    {
      _var_1370 = _var_1366 < _var_1364;
    } break;
    case 1:
    {
      _var_1370 = _var_1364 & 0x1;
    } break;
    case 16:
    {
      _var_1370 = _var_1366 > ~_var_1364;
    } break;
    case 8:
    {
      _var_1370 = _var_1364 > _var_1366;
    } break;
  }
  generic32_t _var_1371;
  generic64_t _var_1372;
  generic32_t _var_1373;
  generic32_t _var_1374;
  generic32_t _var_1375;
  generic64_t _var_1376;
  generic64_t _var_1377;
  _var_1373 = !(number32_t) ((uint64_t) _var_1369 - 1) ? 64 : 0;
  _var_1372 = _lshift(((uint64_t) _var_1369 - 1) & 0xFFFFFFFF, 4294967272);
  _var_1371 = (number32_t) ((uint64_t) _var_1369 - 1) == 2147483647 ? 2048 : 0;
  if (!((((uint32_t) (((_llvm_ctpop_i32((number32_t) ((uint64_t) _var_1369 - 1) & 0xFF) << 2) & 0x4) | _var_1370 | ((((number8_t) ((uint64_t) _var_1369 - 1) + '\001') ^ (number8_t) ((uint64_t) _var_1369 - 1)) & 0x10) | _var_1373 | ((number32_t) _var_1372 & 0x80) | _var_1371) >> 4) ^ (((_llvm_ctpop_i32((number32_t) ((uint64_t) _var_1369 - 1) & 0xFF) << 2) & 0x4) | _var_1370 | ((((number8_t) ((uint64_t) _var_1369 - 1) + '\001') ^ (number8_t) ((uint64_t) _var_1369 - 1)) & 0x10) | _var_1373 | ((number32_t) _var_1372 & 0x80))) & 0xC0)) {
    generic64_t _var_1378;
    float128_t _var_1379;
    _stack.offset_8 = ((uint64_t) _var_1369 - 1) & 0xFFFFFFFF;
    _stack.offset_0 = ((uint64_t) _var_1369 - 1) & 0xFFFFFFFF;
    _helper_fstt_ST0_wrapper(NULL, &_stack, _var_1170, _var_1179, _var_1180, _var_1181, _var_1182, _var_1183, _var_1184, _var_1185, _var_1186, _var_1187, _var_1188, _var_1189, _var_1190, _var_1191, _var_1192, _var_1193, _var_1194);
    _helper_fpop_wrapper(NULL, _var_1170, &_var_1161, &_var_1162, &_var_1163, &_var_1164, &_var_1165, &_var_1166, &_var_1167, &_var_1168, &_var_1169);
    _var_1379 = frexpl((float128_t) ((number128_t) y & ((uint128_t) 0xFFFFFFFFFFFFFFFF)), (int32_t *) &_stack.offset_88);
    _var_1378 = _stack.offset_8;
    _helper_fmov_FT0_STN_wrapper(NULL, 0, _var_1161, _var_1179, _var_1180, _var_1181, _var_1182, _var_1183, _var_1184, _var_1185, _var_1186, _var_1187, _var_1188, _var_1189, _var_1190, _var_1191, _var_1192, _var_1193, _var_1194, &_var_1146, &_var_1147);
    _helper_fadd_ST0_FT0_wrapper(NULL, _var_1161, _var_1179, _var_1180, _var_1181, _var_1182, _var_1183, _var_1184, _var_1185, _var_1186, _var_1187, _var_1188, _var_1189, _var_1190, _var_1191, _var_1192, _var_1193, _var_1194, '\000', '\000', '\000', 'P', '\000', '\000', _var_1146, _var_1147, &_var_1129, &_var_1130, &_var_1131, &_var_1132, &_var_1133, &_var_1134, &_var_1135, &_var_1136, &_var_1137, &_var_1138, &_var_1139, &_var_1140, &_var_1141, &_var_1142, &_var_1143, &_var_1144, &_var_1145);
    _helper_fpush_wrapper(NULL, _var_1161, &_var_1120, &_var_1121, &_var_1122, &_var_1123, &_var_1124, &_var_1125, &_var_1126, &_var_1127, &_var_1128);
    _helper_fldz_ST0_wrapper(NULL, _var_1120, &_var_1104, &_var_1105, &_var_1106, &_var_1107, &_var_1108, &_var_1109, &_var_1110, &_var_1111, &_var_1112, &_var_1113, &_var_1114, &_var_1115, &_var_1116, &_var_1117, &_var_1118, &_var_1119);
    _helper_fxchg_ST0_STN_wrapper(NULL, 1, _var_1120, _var_1104, _var_1105, _var_1106, _var_1107, _var_1108, _var_1109, _var_1110, _var_1111, _var_1112, _var_1113, _var_1114, _var_1115, _var_1116, _var_1117, _var_1118, _var_1119, &_var_1088, &_var_1089, &_var_1090, &_var_1091, &_var_1092, &_var_1093, &_var_1094, &_var_1095, &_var_1096, &_var_1097, &_var_1098, &_var_1099, &_var_1100, &_var_1101, &_var_1102, &_var_1103);
    _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_1120, _var_1088, _var_1089, _var_1090, _var_1091, _var_1092, _var_1093, _var_1094, _var_1095, _var_1096, _var_1097, _var_1098, _var_1099, _var_1100, _var_1101, _var_1102, _var_1103, &_var_1086, &_var_1087);
    _helper_fucomi_ST0_FT0_wrapper(NULL, (uint64_t) _var_1369 - 1, 1, (int64_t) ((((_llvm_ctpop_i32((number32_t) ((uint64_t) _var_1369 - 1) & 0xFF) << 2) & 0x4) | _var_1370 | ((((number8_t) ((uint64_t) _var_1369 - 1) + '\001') ^ (number8_t) ((uint64_t) _var_1369 - 1)) & 0x10) | _var_1373 | ((number32_t) _var_1372 & 0x80) | _var_1371) ^ 0x4), 0, _var_1120, _var_1088, _var_1089, _var_1090, _var_1091, _var_1092, _var_1093, _var_1094, _var_1095, _var_1096, _var_1097, _var_1098, _var_1099, _var_1100, _var_1101, _var_1102, _var_1103, _var_1145, _var_1086, _var_1087, &_var_1084, &_var_1085);
    if ((_var_1084 & 0x44) != 64) {
      _stack.offset_88 = _stack.offset_88 - 1;
    }
    generic32_t _var_1380;
    _var_1380 = _stack.offset_24.member_1 | 0x20;
    if (_var_1380 != 97) {
      generic64_t _var_1381;
      generic64_t _var_1382;
      generic16_t _var_1383;
      generic64_t _var_1384;
      generic16_t _var_1385;
      generic64_t _var_1386;
      generic16_t _var_1387;
      generic64_t _var_1388;
      generic16_t _var_1389;
      generic64_t _var_1390;
      generic16_t _var_1391;
      generic64_t _var_1392;
      generic16_t _var_1393;
      generic64_t _var_1394;
      generic16_t _var_1395;
      generic64_t _var_1396;
      generic16_t _var_1397;
      generic8_t _var_1398;
      _var_1381 = p > -1 ? (uint64_t) p : 6;
      _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_1120, _var_1088, _var_1089, _var_1090, _var_1091, _var_1092, _var_1093, _var_1094, _var_1095, _var_1096, _var_1097, _var_1098, _var_1099, _var_1100, _var_1101, _var_1102, _var_1103, &_var_1031, &_var_1032);
      _helper_fucomi_ST0_FT0_wrapper(NULL, (uint64_t) p, 24, 97, 0, _var_1120, _var_1088, _var_1089, _var_1090, _var_1091, _var_1092, _var_1093, _var_1094, _var_1095, _var_1096, _var_1097, _var_1098, _var_1099, _var_1100, _var_1101, _var_1102, _var_1103, _var_1085, _var_1031, _var_1032, &_var_1029, &_var_1030);
      _var_1398 = _var_1030;
      _var_1382 = _var_1088;
      _var_1383 = _var_1089;
      _var_1384 = _var_1090;
      _var_1385 = _var_1091;
      _var_1386 = _var_1092;
      _var_1387 = _var_1093;
      _var_1388 = _var_1094;
      _var_1389 = _var_1095;
      _var_1390 = _var_1096;
      _var_1391 = _var_1097;
      _var_1392 = _var_1098;
      _var_1393 = _var_1099;
      _var_1394 = _var_1100;
      _var_1395 = _var_1101;
      _var_1396 = _var_1102;
      _var_1397 = _var_1103;
      if ((_var_1029 & 0x44) != 64) {
        _helper_flds_FT0_wrapper(NULL, *((generic32_t *) ""), _var_1030, '\000', '\000', &_var_975, &_var_976, &_var_977);
        _helper_fmul_ST0_FT0_wrapper(NULL, _var_1120, _var_1088, _var_1089, _var_1090, _var_1091, _var_1092, _var_1093, _var_1094, _var_1095, _var_1096, _var_1097, _var_1098, _var_1099, _var_1100, _var_1101, _var_1102, _var_1103, '\000', '\000', _var_975, 'P', '\000', '\000', _var_976, _var_977, &_var_958, &_var_959, &_var_960, &_var_961, &_var_962, &_var_963, &_var_964, &_var_965, &_var_966, &_var_967, &_var_968, &_var_969, &_var_970, &_var_971, &_var_972, &_var_973, &_var_974);
        _var_1382 = _var_958;
        _var_1383 = _var_959;
        _var_1384 = _var_960;
        _var_1385 = _var_961;
        _var_1386 = _var_962;
        _var_1387 = _var_963;
        _var_1388 = _var_964;
        _var_1389 = _var_965;
        _var_1390 = _var_966;
        _var_1391 = _var_967;
        _var_1392 = _var_968;
        _var_1393 = _var_969;
        _var_1394 = _var_970;
        _var_1395 = _var_971;
        _var_1396 = _var_972;
        _var_1397 = _var_973;
        _var_1398 = _var_974;
        _stack.offset_88 = _stack.offset_88 - 28;
      }
      generic32_t _var_1399;
      generic32_t _var_1400;
      generic64_t _var_1401;
      generic64_t _var_1402;
      generic64_t _var_1403;
      generic32_t _var_1404;
      generic16_t _var_1405;
      generic64_t _var_1406;
      generic16_t _var_1407;
      generic64_t _var_1408;
      generic16_t _var_1409;
      generic64_t _var_1410;
      generic16_t _var_1411;
      generic64_t _var_1412;
      generic16_t _var_1413;
      generic64_t _var_1414;
      generic16_t _var_1415;
      generic64_t _var_1416;
      generic16_t _var_1417;
      generic64_t _var_1418;
      generic16_t _var_1419;
      generic64_t _var_1420;
      generic16_t _var_1421;
      generic8_t _var_1422;
      _var_1406 = _var_1382;
      _var_1407 = _var_1383;
      _var_1408 = _var_1384;
      _var_1409 = _var_1385;
      _var_1410 = _var_1386;
      _var_1411 = _var_1387;
      _var_1412 = _var_1388;
      _var_1413 = _var_1389;
      _var_1414 = _var_1390;
      _var_1415 = _var_1391;
      _var_1416 = _var_1392;
      _var_1417 = _var_1393;
      _var_1418 = _var_1394;
      _var_1419 = _var_1395;
      _var_1420 = _var_1396;
      _var_1421 = _var_1397;
      _var_1422 = _var_1398;
      _var_1399 = _stack.offset_88;
      _var_1401 = (int32_t) _var_1399 < (int32_t) 0 ? 18446744073709544220U : 18446744073709551300U;
      _var_1400 = _helper_fnstcw_wrapper(NULL, 895);
      _stack.offset_78 = (number16_t) _var_1400;
      _stack.offset_76 = (number16_t) _var_1400 | 0xC00;
      _var_1403 = (pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1;
      _var_1402 = 0;
      _var_1404 = _var_1120;
      _var_1405 = 895;
      generic64_t _var_1423;
      generic64_t _var_1424;
      generic64_t _var_1425;
      do {
        _var_1424 = _var_1402;
        _var_1423 = _var_1403;
        _helper_fpush_wrapper(NULL, _var_1404, &_var_805, &_var_806, &_var_807, &_var_808, &_var_809, &_var_810, &_var_811, &_var_812, &_var_813);
        _helper_fmov_ST0_STN_wrapper(NULL, 1, _var_805, _var_1406, _var_1407, _var_1408, _var_1409, _var_1410, _var_1411, _var_1412, _var_1413, _var_1414, _var_1415, _var_1416, _var_1417, _var_1418, _var_1419, _var_1420, _var_1421, &_var_789, &_var_790, &_var_791, &_var_792, &_var_793, &_var_794, &_var_795, &_var_796, &_var_797, &_var_798, &_var_799, &_var_800, &_var_801, &_var_802, &_var_803, &_var_804);
        _helper_fldcw_wrapper(NULL, (uint32_t) _stack.offset_76, _var_1405, &_var_786, &_var_787, &_var_788);
        _var_1425 = _helper_fistll_ST0_wrapper(NULL, _var_805, _var_789, _var_790, _var_791, _var_792, _var_793, _var_794, _var_795, _var_796, _var_797, _var_798, _var_799, _var_800, _var_801, _var_802, _var_803, _var_804, _var_787, _var_1422, &_var_785);
        *((generic64_t *) &_stack.offset_56) = _var_1425;
        _helper_fpop_wrapper(NULL, _var_805, &_var_776, &_var_777, &_var_778, &_var_779, &_var_780, &_var_781, &_var_782, &_var_783, &_var_784);
        _helper_fldcw_wrapper(NULL, (uint32_t) _stack.offset_78, _var_786, &_var_773, &_var_774, &_var_775);
        _var_1405 = _var_773;
        _var_1403 = _var_1423 + 4;
        *((generic32_t *) _var_1423) = (number32_t) *((generic64_t *) &_stack.offset_56);
        *((generic64_t *) &_stack.offset_56) = *((generic64_t *) &_stack.offset_56) & 0xFFFFFFFF;
        _helper_fildll_ST0_wrapper(NULL, *((generic64_t *) &_stack.offset_56) & 0xFFFFFFFF, _var_776, &_var_748, &_var_749, &_var_750, &_var_751, &_var_752, &_var_753, &_var_754, &_var_755, &_var_756, &_var_757, &_var_758, &_var_759, &_var_760, &_var_761, &_var_762, &_var_763, &_var_764, &_var_765, &_var_766, &_var_767, &_var_768, &_var_769, &_var_770, &_var_771, &_var_772);
        _helper_fsub_STN_ST0_wrapper(NULL, 1, _var_748, _var_757, _var_758, _var_759, _var_760, _var_761, _var_762, _var_763, _var_764, _var_765, _var_766, _var_767, _var_768, _var_769, _var_770, _var_771, _var_772, '\000', _var_774, _var_785, _var_775, '\000', '\000', &_var_731, &_var_732, &_var_733, &_var_734, &_var_735, &_var_736, &_var_737, &_var_738, &_var_739, &_var_740, &_var_741, &_var_742, &_var_743, &_var_744, &_var_745, &_var_746, &_var_747);
        _helper_fpop_wrapper(NULL, _var_748, &_var_722, &_var_723, &_var_724, &_var_725, &_var_726, &_var_727, &_var_728, &_var_729, &_var_730);
        _var_1404 = _var_722;
        _helper_flds_FT0_wrapper(NULL, *((generic32_t *) "(knN"), _var_747, '\000', '\000', &_var_719, &_var_720, &_var_721);
        _helper_fmul_ST0_FT0_wrapper(NULL, _var_1404, _var_731, _var_732, _var_733, _var_734, _var_735, _var_736, _var_737, _var_738, _var_739, _var_740, _var_741, _var_742, _var_743, _var_744, _var_745, _var_746, '\000', _var_774, _var_719, _var_775, '\000', '\000', _var_720, _var_721, &_var_702, &_var_703, &_var_704, &_var_705, &_var_706, &_var_707, &_var_708, &_var_709, &_var_710, &_var_711, &_var_712, &_var_713, &_var_714, &_var_715, &_var_716, &_var_717, &_var_718);
        _var_1406 = _var_702;
        _var_1407 = _var_703;
        _var_1408 = _var_704;
        _var_1409 = _var_705;
        _var_1410 = _var_706;
        _var_1411 = _var_707;
        _var_1412 = _var_708;
        _var_1413 = _var_709;
        _var_1414 = _var_710;
        _var_1415 = _var_711;
        _var_1416 = _var_712;
        _var_1417 = _var_713;
        _var_1418 = _var_714;
        _var_1419 = _var_715;
        _var_1420 = _var_716;
        _var_1421 = _var_717;
        _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_1404, _var_1406, _var_1407, _var_1408, _var_1409, _var_1410, _var_1411, _var_1412, _var_1413, _var_1414, _var_1415, _var_1416, _var_1417, _var_1418, _var_1419, _var_1420, _var_1421, &_var_700, &_var_701);
        _helper_fucomi_ST0_FT0_wrapper(NULL, (pointer_or_number64_t) &(&_stack)[1] + 4 + _var_1401 * 1 + (_var_1424 << 2), 9, 4, 0, _var_1404, _var_1406, _var_1407, _var_1408, _var_1409, _var_1410, _var_1411, _var_1412, _var_1413, _var_1414, _var_1415, _var_1416, _var_1417, _var_1418, _var_1419, _var_1420, _var_1421, _var_718, _var_700, _var_701, &_var_698, &_var_699);
        _var_1422 = _var_699;
        _var_1402 = _var_1424 + 1;
      } while ((_var_698 & 0x44) != 64);
      generic8_t _var_1426;
      generic64_t _var_1427;
      generic64_t _var_1428;
      generic64_t _var_1429;
      generic32_t _var_1430;
      _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_722, _var_702, _var_703, _var_704, _var_705, _var_706, _var_707, _var_708, _var_709, _var_710, _var_711, _var_712, _var_713, _var_714, _var_715, _var_716, _var_717, &_var_459, &_var_460, &_var_461, &_var_462, &_var_463, &_var_464, &_var_465, &_var_466, &_var_467, &_var_468, &_var_469, &_var_470, &_var_471, &_var_472, &_var_473, &_var_474);
      _helper_fpop_wrapper(NULL, _var_722, &_var_450, &_var_451, &_var_452, &_var_453, &_var_454, &_var_455, &_var_456, &_var_457, &_var_458);
      _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_450, _var_459, _var_460, _var_461, _var_462, _var_463, _var_464, _var_465, _var_466, _var_467, _var_468, _var_469, _var_470, _var_471, _var_472, _var_473, _var_474, &_var_434, &_var_435, &_var_436, &_var_437, &_var_438, &_var_439, &_var_440, &_var_441, &_var_442, &_var_443, &_var_444, &_var_445, &_var_446, &_var_447, &_var_448, &_var_449);
      _helper_fpop_wrapper(NULL, _var_450, &_var_425, &_var_426, &_var_427, &_var_428, &_var_429, &_var_430, &_var_431, &_var_432, &_var_433);
      _var_1427 = _lshift(_var_1399, 4294967272);
      _var_1426 = !_var_1399 ? '@' : '\000';
      _var_1428 = (pointer_or_number64_t) &(&_stack)[1] + 4 + _var_1424 * 4 + _var_1401 * 1;
      _var_1429 = (pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1;
      _var_1430 = _var_1399;
      if (!(_var_1426 | ((number8_t) _var_1427 & 0x80))) {
        generic64_t _var_1431;
        generic32_t _var_1432;
        generic64_t _var_1433;
        generic64_t _var_1434;
        _var_1433 = (pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1;
        _var_1434 = (pointer_or_number64_t) &(&_stack)[1] + 4 + _var_1424 * 4 + _var_1401 * 1;
        _var_1431 = _var_1399;
        _var_1432 = _var_1399;
        generic64_t _var_1435;
        generic8_t _var_1436;
        generic64_t _var_1437;
        generic64_t _var_1438;
        generic64_t _var_1439;
        do {
          generic64_t _var_1440;
          _var_1438 = (int32_t) _var_1432 > (int32_t) 29 ? 29 : _var_1431;
          _var_1440 = 0;
          if (!(_var_1433 > _var_1434 - 4)) {
            generic64_t _var_1441;
            generic64_t _var_1442;
            generic64_t _var_1443;
            _var_1441 = 0;
            _var_1442 = _var_1434 - 4;
            _var_1443 = 0;
            generic64_t _var_1444;
            generic8_t _var_1445;
            do {
              _var_1444 = (_var_1443 & 0xFFFFFFFF) + (*((generic32_t *) _var_1442) << (_var_1438 & 0x3F));
              _var_1443 = _var_1444 / 1000000000;
              *((generic32_t *) _var_1442) = (number32_t) (_var_1444 % 1000000000);
              _var_1442 = _var_1442 - 4;
              _var_1445 = _var_1433 > _var_1434 - 8 - (_var_1441 << 2);
              _var_1441 = _var_1441 + 1;
            } while (!(_var_1445));
            _var_1440 = _var_1443;
          }
          _var_1439 = _var_1433;
          if ((number32_t) _var_1440) {
            _var_1439 = _var_1433 - 4;
            *((generic32_t *) _var_1439) = (number32_t) _var_1440;
          }
          generic64_t _var_1446;
          generic64_t _var_1447;
          _var_1446 = 0;
          _var_1447 = _var_1434;
          while (true) {
            _var_1435 = _var_1447;
            if (_var_1435 > _var_1439) {
              generic8_t _var_1448;
              _var_1447 = _var_1435 - 4;
              _var_1448 = !*((generic32_t *) (_var_1434 - 4 - (_var_1446 << 2)));
              _var_1446 = _var_1446 + 1;
              if (_var_1448) {
                continue;
              }
            }
            break;
          }
          _var_1432 = _var_1432 - (number32_t) _var_1438;
          _var_1431 = _var_1432;
          _var_1437 = _lshift(_var_1431, 4294967272);
          _var_1436 = !_var_1432 ? '@' : '\000';
        } while (!(_var_1436 | ((number8_t) _var_1437 & 0x80)));
        _var_1428 = _var_1435;
        _var_1429 = _var_1439;
        _var_1430 = _var_1432;
      }
      if (!(_var_1426 | ((number8_t) _var_1427 & 0x80))) {
        _stack.offset_88 = _var_1430;
      }
      generic8_t _var_1449;
      generic64_t _var_1450;
      generic64_t _var_1451;
      generic64_t _var_1452;
      generic32_t _var_1453;
      _var_1450 = (int32_t) (number32_t) _var_1381 > -30 && (int32_t) (number32_t) _var_1381 < (int32_t) 2147483619 ? 0 : 18446744069414584320U;
      _var_1453 = _stack.offset_88;
      *((generic64_t *) &_stack.offset_64) = (int64_t) (((number64_t) ((int64_t) (_var_1450 | ((number32_t) _var_1381 + 29)) / (int64_t) 9) << 32) + 4294967296) >> 30;
      _var_1449 = (int32_t) _var_1453 > -1;
      _var_1451 = _var_1428;
      _var_1452 = _var_1429;
      if (!(_var_1449)) {
        generic32_t _var_1454;
        generic64_t _var_1455;
        generic64_t _var_1456;
        _var_1454 = _stack.offset_88;
        _var_1455 = _var_1429;
        _var_1456 = _var_1428;
        generic64_t _var_1457;
        generic64_t _var_1458;
        generic64_t _var_1459;
        do {
          generic32_t _var_1460;
          generic32_t _var_1461;
          _var_1460 = 1953125;
          _var_1461 = 512;
          _var_1459 = 9;
          if (!((int32_t) _var_1454 < -9)) {
            _var_1459 = 0 - _var_1454;
            _var_1460 = 0x3B9ACA00 >> ((0 - _var_1454) & 0x1F);
            _var_1461 = 0x1 << ((0 - _var_1454) & 0x1F);
          }
          generic64_t _var_1462;
          _stack.offset_56 = _var_1460;
          _var_1462 = 0;
          if (_var_1455 < _var_1456) {
            generic64_t _var_1463;
            generic64_t _var_1464;
            generic32_t _var_1465;
            _var_1463 = 0;
            _var_1464 = _var_1455;
            _var_1465 = 0;
            generic64_t _var_1466;
            generic8_t _var_1467;
            do {
              _var_1466 = _var_1464;
              _var_1464 = _var_1464 + 4;
              *((generic32_t *) _var_1466) = _var_1465 + (number32_t) (*((generic32_t *) _var_1466) >> (_var_1459 & 0x1F));
              _var_1465 = ((_var_1461 - 1) & *((generic32_t *) _var_1466)) * _stack.offset_56;
              _var_1467 = _var_1455 + 4 + (_var_1463 << 2) < _var_1456;
              _var_1463 = _var_1463 + 1;
            } while (_var_1467);
            _var_1462 = ((_var_1461 - 1) & *((generic32_t *) _var_1466)) * _stack.offset_56;
          }
          generic64_t _var_1468;
          _var_1457 = !*((generic32_t *) _var_1455) ? _var_1455 + 4 : _var_1455;
          _var_1468 = _var_1456;
          if (_var_1462) {
            *((generic32_t *) _var_1456) = (number32_t) _var_1462;
            _var_1468 = _var_1456 + 4;
          }
          generic64_t _var_1469;
          _var_1458 = _var_1468;
          _var_1469 = _var_1380 == 102 ? (pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1 : _var_1457;
          if ((int64_t) ((int64_t) (_var_1458 - _var_1469) >> 2) > (int64_t) ((int64_t) (((number64_t) ((int64_t) (_var_1450 | ((number32_t) _var_1381 + 29)) / (int64_t) 9) << 32) + 4294967296) >> 32)) {
            _var_1458 = *((generic64_t *) &_stack.offset_64) + _var_1469;
          }
          _var_1454 = _var_1454 + (number32_t) _var_1459;
          _var_1455 = _var_1457;
        } while (!((int32_t) _var_1454 > -1));
        _var_1451 = _var_1458;
        _var_1452 = _var_1457;
        _var_1453 = _var_1454;
      }
      if (!(_var_1449)) {
        _stack.offset_88 = _var_1453;
      }
      generic64_t _var_1470;
      _var_1470 = 0;
      if (_var_1452 < _var_1451) {
        _var_1470 = ((((pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1 - _var_1452) >> 2) * 9) & 0xFFFFFFFF;
        if (!(*((generic32_t *) _var_1452) < 10)) {
          generic64_t _var_1471;
          generic32_t _var_1472;
          _var_1471 = ((((pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1 - _var_1452) >> 2) * 9) & 0xFFFFFFFF;
          _var_1472 = 10;
          do {
            _var_1472 = _var_1472 * 10;
            _var_1471 = (_var_1471 + 1) & 0xFFFFFFFF;
          } while (!(*((generic32_t *) _var_1452) - _var_1472 > ~_var_1472));
          _var_1470 = _var_1471;
        }
      }
      generic64_t _var_1473;
      generic64_t _var_1474;
      generic64_t _var_1475;
      generic64_t _var_1476;
      _var_1476 = _var_1470;
      _var_1473 = _var_1380 == 102 ? 0 : _var_1476;
      _var_1474 = _var_1451;
      _var_1475 = _var_1452;
      if ((int64_t) ((int64_t) ((number64_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) << 32) >> 32) < (int64_t) (((int64_t) (_var_1451 - ((pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1)) >> 2) * 9 - 9)) {
        generic64_t _var_1477;
        generic64_t _var_1478;
        _var_1477 = (int32_t) (number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) > -147457 && (int32_t) (number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) < (int32_t) 2147336192 ? 0 : 18446744069414584320U;
        _var_1478 = 10;
        if ((((int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) % (int64_t) 9) & 0xFFFFFFFF) != 8) {
          generic64_t _var_1479;
          generic64_t _var_1480;
          _var_1479 = 10;
          _var_1480 = (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) % (int64_t) 9;
          do {
            _var_1480 = (_var_1480 + 1) & 0xFFFFFFFF;
            _var_1479 = (_var_1479 * 10) & 0xFFFFFFFC;
          } while (_var_1480 != 8);
          _var_1478 = _var_1479;
        }
        generic64_t _var_1481;
        generic64_t _var_1482;
        generic64_t _var_1483;
        _var_1481 = _var_1452;
        _var_1482 = (pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532;
        _var_1483 = _var_1470;
        if (!(!(*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478) && _var_1451 != (pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65528)) {
          generic64_t _var_1484;
          generic8_t _var_1485;
          if (!((*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) / _var_1478) & 0x1)) {
            if (((number32_t) _var_1478 == 1000000000 && _var_1452 < (pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532) && ((*((generic8_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65536)) & 0x1))) {
              _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &segment_0x405000_Generic64_3292 + 1232, _var_425, &_var_250, &_var_251, &_var_252, &_var_253, &_var_254, &_var_255, &_var_256, &_var_257, &_var_258, &_var_259, &_var_260, &_var_261, &_var_262, &_var_263, &_var_264, &_var_265, &_var_266, &_var_267, &_var_268, &_var_269, &_var_270, &_var_271, &_var_272, &_var_273, &_var_274);
              _var_1484 = &_var_250;
              _var_1485 = _var_699;
            } else {
              _helper_flds_ST0_wrapper(NULL, *((generic32_t *) ""), _var_425, _var_699, '\000', '\000', &_var_224, &_var_225, &_var_226, &_var_227, &_var_228, &_var_229, &_var_230, &_var_231, &_var_232, &_var_233, &_var_234, &_var_235, &_var_236, &_var_237, &_var_238, &_var_239, &_var_240, &_var_241, &_var_242, &_var_243, &_var_244, &_var_245, &_var_246, &_var_247, &_var_248, &_var_249);
              _var_1485 = _var_249;
              _var_1484 = &_var_224;
            }
          } else {
            _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &segment_0x405000_Generic64_3292 + 1232, _var_425, &_var_250, &_var_251, &_var_252, &_var_253, &_var_254, &_var_255, &_var_256, &_var_257, &_var_258, &_var_259, &_var_260, &_var_261, &_var_262, &_var_263, &_var_264, &_var_265, &_var_266, &_var_267, &_var_268, &_var_269, &_var_270, &_var_271, &_var_272, &_var_273, &_var_274);
            _var_1484 = &_var_250;
            _var_1485 = _var_699;
          }
          generic32_t _var_1486;
          generic64_t _var_1487;
          generic16_t _var_1488;
          generic64_t _var_1489;
          generic16_t _var_1490;
          generic64_t _var_1491;
          generic16_t _var_1492;
          generic64_t _var_1493;
          generic16_t _var_1494;
          generic64_t _var_1495;
          generic16_t _var_1496;
          generic64_t _var_1497;
          generic16_t _var_1498;
          generic64_t _var_1499;
          generic16_t _var_1500;
          generic64_t _var_1501;
          generic16_t _var_1502;
          generic8_t _var_1503;
          if (*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478 < (int32_t) (number32_t) _var_1478 >> 1) {
            _helper_flds_ST0_wrapper(NULL, *((generic32_t *) ""), *((generic32_t *) _var_1484), _var_1485, '\000', '\000', &_var_198, &_var_199, &_var_200, &_var_201, &_var_202, &_var_203, &_var_204, &_var_205, &_var_206, &_var_207, &_var_208, &_var_209, &_var_210, &_var_211, &_var_212, &_var_213, &_var_214, &_var_215, &_var_216, &_var_217, &_var_218, &_var_219, &_var_220, &_var_221, &_var_222, &_var_223);
            _var_1486 = _var_198;
            _var_1487 = _var_207;
            _var_1488 = _var_208;
            _var_1489 = _var_209;
            _var_1490 = _var_210;
            _var_1491 = _var_211;
            _var_1492 = _var_212;
            _var_1493 = _var_213;
            _var_1494 = _var_214;
            _var_1495 = _var_215;
            _var_1496 = _var_216;
            _var_1497 = _var_217;
            _var_1498 = _var_218;
            _var_1499 = _var_219;
            _var_1500 = _var_220;
            _var_1501 = _var_221;
            _var_1502 = _var_222;
            _var_1503 = _var_223;
          } else {
            generic64_t _var_1504;
            generic64_t _var_1505;
            generic32_t _var_1506;
            _var_1505 = _lshift((uint64_t) (((int32_t) (number32_t) _var_1478 >> 1) - (number32_t) (*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478)), 4294967272);
            _var_1504 = _lshift((uint64_t) ((((int32_t) (number32_t) _var_1478 >> 1) ^ (number32_t) (*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478)) & (((int32_t) (number32_t) _var_1478 >> 1) ^ (((int32_t) (number32_t) _var_1478 >> 1) - (number32_t) (*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478)))), 4294967276);
            _var_1506 = *((generic32_t *) _var_1484);
            if ((int32_t) (number32_t) _var_1478 >> 1 == (number32_t) (*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478)) {
              _helper_fpush_wrapper(NULL, *((generic32_t *) _var_1484), &_var_163, &_var_164, &_var_165, &_var_166, &_var_167, &_var_168, &_var_169, &_var_170, &_var_171);
              _var_1486 = _var_163;
              _helper_fld1_ST0_wrapper(NULL, _var_1486, &_var_147, &_var_148, &_var_149, &_var_150, &_var_151, &_var_152, &_var_153, &_var_154, &_var_155, &_var_156, &_var_157, &_var_158, &_var_159, &_var_160, &_var_161, &_var_162);
              _var_1487 = _var_147;
              _var_1488 = _var_148;
              _var_1489 = _var_149;
              _var_1490 = _var_150;
              _var_1491 = _var_151;
              _var_1492 = _var_152;
              _var_1493 = _var_153;
              _var_1494 = _var_154;
              _var_1495 = _var_155;
              _var_1496 = _var_156;
              _var_1497 = _var_157;
              _var_1498 = _var_158;
              _var_1499 = _var_159;
              _var_1500 = _var_160;
              _var_1501 = _var_161;
              _var_1502 = _var_162;
              _var_1503 = _var_1485;
              if (_var_1451 != (pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65528) {
                _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_163, _var_147, _var_148, _var_149, _var_150, _var_151, _var_152, _var_153, _var_154, _var_155, _var_156, _var_157, _var_158, _var_159, _var_160, _var_161, _var_162, &_var_74, &_var_75, &_var_76, &_var_77, &_var_78, &_var_79, &_var_80, &_var_81, &_var_82, &_var_83, &_var_84, &_var_85, &_var_86, &_var_87, &_var_88, &_var_89);
                _helper_fpop_wrapper(NULL, _var_163, &_var_65, &_var_66, &_var_67, &_var_68, &_var_69, &_var_70, &_var_71, &_var_72, &_var_73);
                _var_1506 = _var_65;
                _helper_flds_ST0_wrapper(NULL, *((generic32_t *) ""), _var_1506, _var_1485, '\000', '\000', &_var_172, &_var_173, &_var_174, &_var_175, &_var_176, &_var_177, &_var_178, &_var_179, &_var_180, &_var_181, &_var_182, &_var_183, &_var_184, &_var_185, &_var_186, &_var_187, &_var_188, &_var_189, &_var_190, &_var_191, &_var_192, &_var_193, &_var_194, &_var_195, &_var_196, &_var_197);
                _var_1486 = _var_172;
                _var_1487 = _var_181;
                _var_1488 = _var_182;
                _var_1489 = _var_183;
                _var_1490 = _var_184;
                _var_1491 = _var_185;
                _var_1492 = _var_186;
                _var_1493 = _var_187;
                _var_1494 = _var_188;
                _var_1495 = _var_189;
                _var_1496 = _var_190;
                _var_1497 = _var_191;
                _var_1498 = _var_192;
                _var_1499 = _var_193;
                _var_1500 = _var_194;
                _var_1501 = _var_195;
                _var_1502 = _var_196;
                _var_1503 = _var_197;
              }
            } else {
              _helper_flds_ST0_wrapper(NULL, *((generic32_t *) ""), _var_1506, _var_1485, '\000', '\000', &_var_172, &_var_173, &_var_174, &_var_175, &_var_176, &_var_177, &_var_178, &_var_179, &_var_180, &_var_181, &_var_182, &_var_183, &_var_184, &_var_185, &_var_186, &_var_187, &_var_188, &_var_189, &_var_190, &_var_191, &_var_192, &_var_193, &_var_194, &_var_195, &_var_196, &_var_197);
              _var_1486 = _var_172;
              _var_1487 = _var_181;
              _var_1488 = _var_182;
              _var_1489 = _var_183;
              _var_1490 = _var_184;
              _var_1491 = _var_185;
              _var_1492 = _var_186;
              _var_1493 = _var_187;
              _var_1494 = _var_188;
              _var_1495 = _var_189;
              _var_1496 = _var_190;
              _var_1497 = _var_191;
              _var_1498 = _var_192;
              _var_1499 = _var_193;
              _var_1500 = _var_194;
              _var_1501 = _var_195;
              _var_1502 = _var_196;
              _var_1503 = _var_197;
            }
          }
          generic64_t _var_1507;
          generic16_t _var_1508;
          generic64_t _var_1509;
          generic16_t _var_1510;
          generic64_t _var_1511;
          generic16_t _var_1512;
          generic64_t _var_1513;
          generic16_t _var_1514;
          generic64_t _var_1515;
          generic16_t _var_1516;
          generic64_t _var_1517;
          generic16_t _var_1518;
          generic64_t _var_1519;
          generic16_t _var_1520;
          generic64_t _var_1521;
          generic16_t _var_1522;
          _var_1507 = _var_1487;
          _var_1508 = _var_1488;
          _var_1509 = _var_1489;
          _var_1510 = _var_1490;
          _var_1511 = _var_1491;
          _var_1512 = _var_1492;
          _var_1513 = _var_1493;
          _var_1514 = _var_1494;
          _var_1515 = _var_1495;
          _var_1516 = _var_1496;
          _var_1517 = _var_1497;
          _var_1518 = _var_1498;
          _var_1519 = _var_1499;
          _var_1520 = _var_1500;
          _var_1521 = _var_1501;
          _var_1522 = _var_1502;
          if (_stack.offset_48.member_0) {
            _var_1507 = _var_1487;
            _var_1508 = _var_1488;
            _var_1509 = _var_1489;
            _var_1510 = _var_1490;
            _var_1511 = _var_1491;
            _var_1512 = _var_1492;
            _var_1513 = _var_1493;
            _var_1514 = _var_1494;
            _var_1515 = _var_1495;
            _var_1516 = _var_1496;
            _var_1517 = _var_1497;
            _var_1518 = _var_1498;
            _var_1519 = _var_1499;
            _var_1520 = _var_1500;
            _var_1521 = _var_1501;
            _var_1522 = _var_1502;
            if (*_var_1347 == '-') {
              _helper_fxchg_ST0_STN_wrapper(NULL, 1, _var_1486, _var_1487, _var_1488, _var_1489, _var_1490, _var_1491, _var_1492, _var_1493, _var_1494, _var_1495, _var_1496, _var_1497, _var_1498, _var_1499, _var_1500, _var_1501, _var_1502, &_var_49, &_var_50, &_var_51, &_var_52, &_var_53, &_var_54, &_var_55, &_var_56, &_var_57, &_var_58, &_var_59, &_var_60, &_var_61, &_var_62, &_var_63, &_var_64);
              _helper_fchs_ST0_wrapper(NULL, _var_1486, _var_49, _var_50, _var_51, _var_52, _var_53, _var_54, _var_55, _var_56, _var_57, _var_58, _var_59, _var_60, _var_61, _var_62, _var_63, _var_64, &_var_33, &_var_34, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42, &_var_43, &_var_44, &_var_45, &_var_46, &_var_47, &_var_48);
              _helper_fxchg_ST0_STN_wrapper(NULL, 1, _var_1486, _var_33, _var_34, _var_35, _var_36, _var_37, _var_38, _var_39, _var_40, _var_41, _var_42, _var_43, _var_44, _var_45, _var_46, _var_47, _var_48, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32);
              _helper_fchs_ST0_wrapper(NULL, _var_1486, _var_17, _var_18, _var_19, _var_20, _var_21, _var_22, _var_23, _var_24, _var_25, _var_26, _var_27, _var_28, _var_29, _var_30, _var_31, _var_32, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16);
              _var_1507 = _var_1;
              _var_1508 = _var_2;
              _var_1509 = _var_3;
              _var_1510 = _var_4;
              _var_1511 = _var_5;
              _var_1512 = _var_6;
              _var_1513 = _var_7;
              _var_1514 = _var_8;
              _var_1515 = _var_9;
              _var_1516 = _var_10;
              _var_1517 = _var_11;
              _var_1518 = _var_12;
              _var_1519 = _var_13;
              _var_1520 = _var_14;
              _var_1521 = _var_15;
              _var_1522 = _var_16;
            }
          }
          _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_1486, _var_1507, _var_1508, _var_1509, _var_1510, _var_1511, _var_1512, _var_1513, _var_1514, _var_1515, _var_1516, _var_1517, _var_1518, _var_1519, _var_1520, _var_1521, _var_1522, &_var_145, &_var_146);
          _helper_fadd_ST0_FT0_wrapper(NULL, _var_1486, _var_1507, _var_1508, _var_1509, _var_1510, _var_1511, _var_1512, _var_1513, _var_1514, _var_1515, _var_1516, _var_1517, _var_1518, _var_1519, _var_1520, _var_1521, _var_1522, '\000', _var_774, _var_1503, _var_775, '\000', '\000', _var_145, _var_146, &_var_128, &_var_129, &_var_130, &_var_131, &_var_132, &_var_133, &_var_134, &_var_135, &_var_136, &_var_137, &_var_138, &_var_139, &_var_140, &_var_141, &_var_142, &_var_143, &_var_144);
          _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_1486, _var_128, _var_129, _var_130, _var_131, _var_132, _var_133, _var_134, _var_135, _var_136, _var_137, _var_138, _var_139, _var_140, _var_141, _var_142, _var_143, &_var_126, &_var_127);
          _helper_fucomi_ST0_FT0_wrapper(NULL, *((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) - *((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478, 16, *((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478, 0, _var_1486, _var_128, _var_129, _var_130, _var_131, _var_132, _var_133, _var_134, _var_135, _var_136, _var_137, _var_138, _var_139, _var_140, _var_141, _var_142, _var_143, _var_144, _var_126, _var_127, &_var_124, &_var_125);
          _helper_fpop_wrapper(NULL, _var_1486, &_var_115, &_var_116, &_var_117, &_var_118, &_var_119, &_var_120, &_var_121, &_var_122, &_var_123);
          _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_115, _var_128, _var_129, _var_130, _var_131, _var_132, _var_133, _var_134, _var_135, _var_136, _var_137, _var_138, _var_139, _var_140, _var_141, _var_142, _var_143, &_var_99, &_var_100, &_var_101, &_var_102, &_var_103, &_var_104, &_var_105, &_var_106, &_var_107, &_var_108, &_var_109, &_var_110, &_var_111, &_var_112, &_var_113, &_var_114);
          _helper_fpop_wrapper(NULL, _var_115, &_var_90, &_var_91, &_var_92, &_var_93, &_var_94, &_var_95, &_var_96, &_var_97, &_var_98);
          if ((_var_124 & 0x44) == 64) {
            *((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) = (number32_t) (*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) - *((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478);
            _var_1481 = _var_1452;
            _var_1482 = (pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532;
            _var_1483 = _var_1470;
          } else {
            generic64_t _var_1523;
            generic64_t _var_1524;
            *((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) = (number32_t) _var_1478 + (number32_t) (*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) - *((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478);
            _var_1523 = _var_1452;
            _var_1524 = (pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532;
            if ((number32_t) _var_1478 + (number32_t) (*((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) - *((generic32_t *) ((pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532)) % _var_1478) > 999999999) {
              generic64_t _var_1525;
              generic64_t _var_1526;
              generic64_t _var_1527;
              _var_1525 = 0;
              _var_1526 = (pointer_or_number64_t) &(&_stack)[1] + (int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9 * 4 + _var_1401 * 1 - 65532;
              _var_1527 = _var_1452;
              generic64_t _var_1528;
              generic64_t _var_1529;
              do {
                _var_1529 = _var_1527;
                _var_1528 = _var_1401 + (pointer_or_number64_t) &_stack + (((int64_t) (_var_1477 | ((number32_t) (((_var_1381 - _var_1473) & 0xFFFFFFFF) + (pointer_or_number64_t) (_var_1380 == 103 && _var_1381 != 0)) + 147456)) / (int64_t) 9) << 2) - 58008 - (_var_1525 << 2);
                *((generic32_t *) _var_1526) = 0;
                if (_var_1529 > _var_1528) {
                  _var_1529 = _var_1527 - 4;
                  *((generic32_t *) _var_1529) = 0;
                }
                *((generic32_t *) _var_1528) = *((generic32_t *) _var_1528) + 1;
                _var_1525 = _var_1525 + 1;
                _var_1526 = _var_1526 - 4;
              } while (*((generic32_t *) _var_1528) > 999999998 && *((generic32_t *) _var_1528) < (uint32_t) -1);
              _var_1523 = _var_1529;
              _var_1524 = _var_1528;
            }
            _var_1481 = _var_1523;
            _var_1482 = _var_1524;
            _var_1483 = ((((pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1 - _var_1481) >> 2) * 9) & 0xFFFFFFFF;
            if (!(*((generic32_t *) _var_1481) < 10)) {
              generic64_t _var_1530;
              generic32_t _var_1531;
              _var_1530 = ((((pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1 - _var_1481) >> 2) * 9) & 0xFFFFFFFF;
              _var_1531 = 10;
              do {
                _var_1531 = _var_1531 * 10;
                _var_1530 = (_var_1530 + 1) & 0xFFFFFFFF;
              } while (!(*((generic32_t *) _var_1481) - _var_1531 > ~_var_1531));
              _var_1481 = _var_1523;
              _var_1482 = _var_1524;
              _var_1483 = _var_1530;
            }
          }
        }
        _var_1475 = _var_1481;
        _var_1476 = _var_1483;
        _var_1474 = _llvm_umin_i64(_var_1451, _var_1482 + 4);
      }
      generic64_t _var_1532;
      generic64_t _var_1533;
      generic64_t _var_1534;
      _var_1534 = _var_1474;
      _var_1532 = _var_1534 - 4;
      _var_1533 = 0;
      generic64_t _var_1535;
      while (true) {
        _var_1535 = _var_1534;
        if (_var_1535 > _var_1475) {
          generic8_t _var_1536;
          _var_1534 = _var_1535 - 4;
          _var_1536 = !*((generic32_t *) (_var_1532 - (_var_1533 << 2)));
          _var_1533 = _var_1533 + 1;
          if (_var_1536) {
            continue;
          }
        }
        break;
      }
      generic64_t _var_1537;
      _var_1537 = _var_1381;
      if (_var_1380 == 103) {
        generic64_t _var_1538;
        generic32_t _var_1539;
        generic64_t _var_1540;
        _var_1538 = !_var_1381 ? 1 : _var_1381;
        if ((int64_t) ((number64_t) _var_1538 << 32) <= (int64_t) ((number64_t) _var_1476 << 32) || (int32_t) (number32_t) _var_1476 < -4) {
          _var_1539 = _stack.offset_24.member_1 - 2;
          _stack.offset_24.member_1 = _var_1539;
          _var_1540 = _var_1538 + 4294967295;
        } else {
          _var_1539 = _stack.offset_24.member_1 - 1;
          _stack.offset_24.member_1 = _var_1539;
          _var_1540 = _var_1538 - ((_var_1476 + 1) & 0xFFFFFFFF);
        }
        _var_1537 = _var_1540 & 0xFFFFFFFF;
        if (!(_stack.offset_20 & 0x8)) {
          generic64_t _var_1541;
          _var_1541 = 9;
          if (_var_1535 > _var_1475) {
            _var_1541 = 9;
            if (*((generic32_t *) (_var_1535 - 4))) {
              _var_1541 = 0;
              if (!(*((generic32_t *) (_var_1535 - 4)) % 10)) {
                generic64_t _var_1542;
                generic64_t _var_1543;
                _var_1542 = 10;
                _var_1543 = 0;
                generic64_t _var_1544;
                do {
                  _var_1542 = (_var_1542 * 10) & 0xFFFFFFFC;
                  _var_1544 = _var_1543 + 1;
                  _var_1543 = _var_1544 & 0xFFFFFFFF;
                } while (!(*((generic32_t *) (_var_1535 - 4)) % _var_1542));
                _var_1541 = (int64_t) ((number64_t) _var_1544 << 32) >> 32;
              }
            }
          }
          generic64_t _var_1545;
          _var_1545 = (_var_1539 & 0xFFFFFFDF) == 70 ? 0 : (int64_t) ((number64_t) _var_1476 << 32) >> 32;
          _var_1537 = _llvm_smin_i64(_llvm_smax_i64(((int64_t) (_var_1535 - ((pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1)) >> 2) * 9 - 9 + _var_1545 - _var_1541, 0), (int64_t) ((number64_t) _var_1540 << 32) >> 32);
        }
      }
      generic64_t _var_1546;
      _stack.offset_56 = *((generic32_t *) &_stack.offset_20) & 0x8;
      _stack.offset_72 = (*((generic32_t *) &_stack.offset_20) & 0x8) | (number32_t) _var_1537;
      _stack.offset_64 = _stack.offset_24.member_1 | 0x20;
      if (_stack.offset_64 == 102) {
        generic64_t _var_1547;
        generic8_t _var_1548;
        generic64_t _var_1549;
        _var_1549 = _lshift(_var_1476 & 0xFFFFFFFF, 4294967272);
        _var_1548 = !(_var_1476 & 0xFFFFFFFF) ? '@' : '\000';
        _var_1547 = !(_var_1548 | ((number8_t) _var_1549 & 0x80)) ? _var_1476 : 0;
        _var_1546 = _var_1547;
      } else {
        generic8_t _var_1550;
        generic64_t _var_1551;
        generic8_t _var_1552;
        int8_t *_var_1553;
        generic64_t _var_1554;
        generic64_t _var_1555;
        _var_1554 = !(_var_1476 & 0x80000000) ? 0 : 4294967295;
        _var_1553 = fmt_u((int64_t) ((number64_t) ((_var_1554 ^ _var_1476) - _var_1554) << 32) >> 32, (int8_t *) &_stack.offset_107.member_0.member_0.member_1);
        _var_1555 = _var_1553;
        _var_1552 = !((pointer_or_number64_t) &_stack.offset_107 + ~_var_1555 * 1) ? '@' : '\000';
        _var_1551 = _lshift((pointer_or_number64_t) &_stack.offset_107 + ~_var_1555, 4294967240);
        _var_1550 = (pointer_or_number64_t) &_stack.offset_107 + ~_var_1555 * 1 == 9223372036854775807 ? '\200' : '\000';
        if ((_var_1552 | ((number8_t) _var_1551 & 0x80)) != _var_1550) {
          generic64_t _var_1556;
          generic64_t _var_1557;
          _var_1556 = 0;
          _var_1557 = _var_1553;
          generic64_t _var_1558;
          generic64_t _var_1559;
          generic8_t _var_1560;
          generic64_t _var_1561;
          generic8_t _var_1562;
          do {
            _var_1558 = _var_1557;
            _var_1559 = (pointer_or_number64_t) _var_1553 - 1 - _var_1556;
            _var_1557 = _var_1558 - 1;
            *((generic8_t *) _var_1559) = '0';
            _var_1562 = (pointer_or_number64_t) &_stack.offset_107 == _var_1558 ? '@' : '\000';
            _var_1561 = _lshift((pointer_or_number64_t) &_stack.offset_107 - (number64_t) _var_1553 + _var_1556, 4294967240);
            _var_1560 = (pointer_or_number64_t) &_stack.offset_107 - (number64_t) _var_1553 + _var_1556 == 9223372036854775807 ? '\200' : '\000';
            _var_1556 = _var_1556 + 1;
          } while ((_var_1562 | ((number8_t) _var_1561 & 0x80)) != _var_1560);
          _var_1555 = _var_1559;
        }
        *((generic64_t *) &_stack.offset_32) = _var_1555 - 2;
        *((generic8_t *) (_var_1555 - 1)) = ((number8_t) ((uint64_t) _var_1476 >> 30) & 0x2) + '+';
        *((generic8_t *) (_var_1555 - 2)) = _stack.offset_24.member_0;
        _var_1546 = (pointer_or_number64_t) &(&_stack)[570532].offset_107 + 2400 - *((generic64_t *) &_stack.offset_32);
      }
      generic32_t _var_1563;
      _var_1563 = _stack.offset_16;
      _stack.offset_24.member_1 = _stack.offset_48.member_0 + (number32_t) (_var_1537 + (((*((generic32_t *) &_stack.offset_20) & 0x8) | (number32_t) _var_1537) != 0) + 1 + _var_1546);
      pad(f, (int8_t) 32, (int32_t) _var_1563, (int32_t) _stack.offset_24.member_1, (int32_t) *((generic32_t *) &_stack.offset_20));
      out(f, (const int8_t *) _var_1347, (size_t) _stack.offset_48.member_0);
      pad(f, (int8_t) 48, (int32_t) _stack.offset_16, (int32_t) _stack.offset_24.member_1, (int32_t) (*((generic32_t *) &_stack.offset_20) ^ 0x10000));
      if (_stack.offset_64 == 102) {
        generic64_t _var_1564;
        generic64_t _var_1565;
        generic64_t _var_1566;
        _var_1566 = _llvm_umin_i64((pointer_or_number64_t) &(&_stack)[1] + _var_1401, _var_1475);
        _var_1564 = _var_1566 + 4;
        _var_1565 = 0;
        generic8_t _var_1567;
        generic64_t _var_1568;
        do {
          int8_t *_var_1569;
          _var_1569 = fmt_u(*((generic32_t *) _var_1566), (int8_t *) &_stack.offset_107.member_2.offset_9.member_0.member_1);
          if (_var_1566 == _llvm_umin_i64((pointer_or_number64_t) &(&_stack)[1] + _var_1401, _var_1475)) {
            _var_1568 = _var_1569;
            if ((pointer_or_number64_t) _var_1569 == (pointer_or_number64_t) &_stack.offset_107.member_2.offset_9) {
              _stack.offset_107.member_1.offset_8.member_0.member_1 = '0';
              _var_1568 = &_stack.offset_107.member_1.offset_8;
            }
          } else {
            if ((uint64_t) _var_1569 > (uint64_t) &_stack.offset_107) {
              generic64_t _var_1570;
              _var_1570 = 0;
              generic8_t _var_1571;
              do {
                *((generic8_t *) ((pointer_or_number64_t) _var_1569 - 1 - _var_1570)) = '0';
                _var_1571 = (pointer_or_number64_t) _var_1569 - 1 - _var_1570 > (uint64_t) &_stack.offset_107;
                _var_1570 = _var_1570 + 1;
              } while (_var_1571);
            }
            _var_1568 = _llvm_umin_i64(_var_1569, (pointer_or_number64_t) &_stack + 107);
          }
          _var_1566 = _var_1566 + 4;
          out(f, (const int8_t *) _var_1568, (pointer_or_number64_t) &_stack.offset_107.member_2.offset_9 - _var_1568);
          _var_1567 = _var_1564 + (_var_1565 << 2) > (pointer_or_number64_t) &(&_stack)[1] + _var_1401 * 1;
          _var_1565 = _var_1565 + 1;
        } while (!(_var_1567));
        generic64_t _var_1572;
        _var_1572 = 0;
        if (_stack.offset_72) {
          generic64_t _var_1573;
          _var_1573 = (pointer_or_number64_t) &(&_stack)[1] + 1 + _var_1401 * 1 < _llvm_umin_i64((pointer_or_number64_t) &(&_stack)[1] + _var_1401, _var_1475) - 3 ? 0 : ((pointer_or_number64_t) &(&_stack)[1] + 4 + _var_1401 * 1 - _llvm_umin_i64((pointer_or_number64_t) &(&_stack)[1] + _var_1401, _var_1475)) & 0xFFFFFFFFFFFFFFFC;
          out(f, (const int8_t *) ".", 1);
          _var_1572 = _var_1537;
          if (_var_1573 + _llvm_umin_i64((pointer_or_number64_t) &(&_stack)[1] + _var_1401, _var_1475) < _var_1535) {
            generic64_t _var_1574;
            generic64_t _var_1575;
            generic64_t _var_1576;
            _var_1574 = 0;
            _var_1575 = _var_1537;
            _var_1576 = _var_1573 + _llvm_umin_i64((pointer_or_number64_t) &(&_stack)[1] + _var_1401, _var_1475);
            generic64_t _var_1577;
            while (true) {
              generic8_t _var_1578;
              generic64_t _var_1579;
              _var_1577 = _var_1575;
              _var_1579 = _lshift(_var_1577 & 0xFFFFFFFF, 4294967272);
              _var_1578 = !(number32_t) _var_1577 ? '@' : '\000';
              if (!(_var_1578 | ((number8_t) _var_1579 & 0x80))) {
                int8_t *_var_1580;
                _var_1580 = fmt_u(*((generic32_t *) _var_1576), (int8_t *) &_stack.offset_107.member_2.offset_9.member_0.member_1);
                if ((uint64_t) _var_1580 > (uint64_t) &_stack.offset_107) {
                  generic64_t _var_1581;
                  _var_1581 = 0;
                  generic8_t _var_1582;
                  do {
                    *((generic8_t *) ((pointer_or_number64_t) _var_1580 - 1 - _var_1581)) = '0';
                    _var_1582 = (pointer_or_number64_t) _var_1580 - 1 - _var_1581 > (uint64_t) &_stack.offset_107;
                    _var_1581 = _var_1581 + 1;
                  } while (_var_1582);
                }
                generic8_t _var_1583;
                generic64_t _var_1584;
                _var_1584 = (int32_t) (number32_t) _var_1577 > (int32_t) 9 ? 9 : (int64_t) ((number64_t) _var_1575 << 32) >> 32;
                _var_1575 = (_var_1575 + 4294967287) & 0xFFFFFFFF;
                _var_1577 = _var_1575;
                _var_1576 = _var_1576 + 4;
                out(f, (const int8_t *) _llvm_umin_i64(_var_1580, (pointer_or_number64_t) &_stack + 107), _var_1584);
                _var_1583 = _var_1573 + _llvm_umin_i64((pointer_or_number64_t) &(&_stack)[1] + _var_1401, _var_1475) + 4 + (_var_1574 << 2) < _var_1535;
                _var_1574 = _var_1574 + 1;
                if (_var_1583) {
                  continue;
                }
              }
              break;
            }
            _var_1572 = _var_1577;
          }
        }
        pad(f, (int8_t) 48, (int32_t) ((number32_t) _var_1572 + 9), (int32_t) 9, (int32_t) 0);
      } else {
        generic64_t _var_1585;
        generic64_t _var_1586;
        _var_1585 = _var_1535 > _var_1475 ? _var_1535 : _var_1475 + 4;
        _var_1586 = _var_1537;
        if (_var_1475 < _var_1585 && (int32_t) (number32_t) _var_1537 > -1) {
          generic64_t _var_1587;
          generic32_t _var_1588;
          generic64_t _var_1589;
          generic64_t _var_1590;
          _var_1587 = 0;
          _var_1588 = (number32_t) _var_1537;
          _var_1589 = _var_1537;
          _var_1590 = _var_1475;
          generic64_t _var_1591;
          generic8_t _var_1592;
          do {
            int8_t *_var_1593;
            generic64_t _var_1594;
            _var_1593 = fmt_u(*((generic32_t *) _var_1590), (int8_t *) &_stack.offset_107.member_2.offset_9.member_0.member_1);
            _var_1594 = _var_1593;
            if (_var_1594 == (pointer_or_number64_t) &_stack.offset_107.member_2.offset_9) {
              _stack.offset_107.member_1.offset_8.member_0.member_1 = '0';
              _var_1594 = &_stack.offset_107.member_1.offset_8;
            }
            if (_var_1590 == _var_1475) {
              *((int8_t **) &_stack.offset_48) = &((int8_t *) _var_1594)[1];
              out(f, (const int8_t *) _var_1594, 1);
              if ((_stack.offset_56 | _var_1588)) {
                out(f, (const int8_t *) ".", 1);
              }
            } else {
              if (_var_1594 > (uint64_t) &_stack.offset_107) {
                generic64_t _var_1595;
                _var_1595 = 0;
                generic8_t _var_1596;
                do {
                  *((generic8_t *) (_var_1594 - 1 - _var_1595)) = '0';
                  _var_1596 = _var_1594 - 1 - _var_1595 > (uint64_t) &_stack.offset_107;
                  _var_1595 = _var_1595 + 1;
                } while (_var_1596);
              }
              *((generic64_t *) &_stack.offset_48) = _llvm_umin_i64(_var_1594, (pointer_or_number64_t) &_stack + 107);
            }
            *((generic64_t *) &_stack.offset_64) = (pointer_or_number64_t) &_stack.offset_107.member_2.offset_9 - *((generic64_t *) &_stack.offset_48);
            _var_1590 = _var_1590 + 4;
            out(f, (const int8_t *) *((generic64_t *) &_stack.offset_48), _llvm_smin_i64((int64_t) ((number64_t) _var_1589 << 32) >> 32, (pointer_or_number64_t) &_stack.offset_107.member_2.offset_9 - *((generic64_t *) &_stack.offset_48)));
            _var_1591 = _var_1589 - *((generic64_t *) &_stack.offset_64);
            _var_1589 = _var_1591 & 0xFFFFFFFF;
            _var_1588 = (number32_t) _var_1591;
            _var_1592 = _var_1475 + 4 + (_var_1587 << 2) < _var_1585 && (int32_t) _var_1588 > -1;
            _var_1587 = _var_1587 + 1;
          } while (_var_1592);
          _var_1586 = _var_1591 & 0xFFFFFFFF;
        }
        pad(f, (int8_t) 48, (int32_t) ((number32_t) _var_1586 + 18), (int32_t) 18, (int32_t) 0);
        out(f, (const int8_t *) *((generic64_t *) &_stack.offset_32), (pointer_or_number64_t) &_stack.offset_107 - *((generic64_t *) &_stack.offset_32));
      }
      pad(f, (int8_t) 32, (int32_t) _stack.offset_16, (int32_t) _stack.offset_24.member_1, (int32_t) (*((generic32_t *) &_stack.offset_20) ^ 0x2000));
      _var_1374 = _llvm_smax_i32(_stack.offset_16, _stack.offset_24.member_1);
      return (int32_t) _var_1374;
    }
    uint8_t *_var_1597;
    generic32_t _var_1598;
    generic64_t _var_1599;
    generic16_t _var_1600;
    generic64_t _var_1601;
    generic16_t _var_1602;
    generic64_t _var_1603;
    generic16_t _var_1604;
    generic64_t _var_1605;
    generic16_t _var_1606;
    generic64_t _var_1607;
    generic16_t _var_1608;
    generic64_t _var_1609;
    generic16_t _var_1610;
    generic64_t _var_1611;
    generic16_t _var_1612;
    generic64_t _var_1613;
    generic16_t _var_1614;
    generic8_t _var_1615;
    _helper_fmov_STN_ST0_wrapper(NULL, 1, _var_1120, _var_1088, _var_1089, _var_1090, _var_1091, _var_1092, _var_1093, _var_1094, _var_1095, _var_1096, _var_1097, _var_1098, _var_1099, _var_1100, _var_1101, _var_1102, _var_1103, &_var_1068, &_var_1069, &_var_1070, &_var_1071, &_var_1072, &_var_1073, &_var_1074, &_var_1075, &_var_1076, &_var_1077, &_var_1078, &_var_1079, &_var_1080, &_var_1081, &_var_1082, &_var_1083);
    _helper_fpop_wrapper(NULL, _var_1120, &_var_1059, &_var_1060, &_var_1061, &_var_1062, &_var_1063, &_var_1064, &_var_1065, &_var_1066, &_var_1067);
    _var_1597 = !(_stack.offset_24.member_0 & 0x20) ? (generic64_t) _var_1347 : (generic64_t) &_var_1347[9];
    _helper_flds_ST0_wrapper(NULL, *((generic32_t *) ""), _var_1059, _var_1085, '\000', '\000', &_var_1033, &_var_1034, &_var_1035, &_var_1036, &_var_1037, &_var_1038, &_var_1039, &_var_1040, &_var_1041, &_var_1042, &_var_1043, &_var_1044, &_var_1045, &_var_1046, &_var_1047, &_var_1048, &_var_1049, &_var_1050, &_var_1051, &_var_1052, &_var_1053, &_var_1054, &_var_1055, &_var_1056, &_var_1057, &_var_1058);
    _stack.offset_48.member_0 = _stack.offset_48.member_0 + 2;
    if ((uint32_t) p > 14) {
      _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_1033, _var_1042, _var_1043, _var_1044, _var_1045, _var_1046, _var_1047, _var_1048, _var_1049, _var_1050, _var_1051, _var_1052, _var_1053, _var_1054, _var_1055, _var_1056, _var_1057, &_var_1013, &_var_1014, &_var_1015, &_var_1016, &_var_1017, &_var_1018, &_var_1019, &_var_1020, &_var_1021, &_var_1022, &_var_1023, &_var_1024, &_var_1025, &_var_1026, &_var_1027, &_var_1028);
      _var_1599 = _var_1013;
      _var_1600 = _var_1014;
      _var_1601 = _var_1015;
      _var_1602 = _var_1016;
      _var_1603 = _var_1017;
      _var_1604 = _var_1018;
      _var_1605 = _var_1019;
      _var_1606 = _var_1020;
      _var_1607 = _var_1021;
      _var_1608 = _var_1022;
      _var_1609 = _var_1023;
      _var_1610 = _var_1024;
      _var_1611 = _var_1025;
      _var_1612 = _var_1026;
      _var_1613 = _var_1027;
      _var_1614 = _var_1028;
      _helper_fpop_wrapper(NULL, _var_1033, &_var_1004, &_var_1005, &_var_1006, &_var_1007, &_var_1008, &_var_1009, &_var_1010, &_var_1011, &_var_1012);
      _var_1598 = _var_1004;
      _var_1615 = _var_1058;
    } else {
      generic64_t _var_1616;
      generic64_t _var_1617;
      generic64_t _var_1618;
      generic64_t _var_1619;
      generic64_t _var_1620;
      generic64_t _var_1621;
      generic64_t _var_1622;
      generic64_t _var_1623;
      generic64_t _var_1624;
      generic64_t _var_1625;
      generic64_t _var_1626;
      generic64_t _var_1627;
      generic64_t _var_1628;
      generic64_t _var_1629;
      generic64_t _var_1630;
      generic64_t _var_1631;
      generic64_t _var_1632;
      generic64_t _var_1633;
      _helper_flds_ST0_wrapper(NULL, *((generic32_t *) ""), _var_1033, _var_1058, '\000', '\000', &_var_978, &_var_979, &_var_980, &_var_981, &_var_982, &_var_983, &_var_984, &_var_985, &_var_986, &_var_987, &_var_988, &_var_989, &_var_990, &_var_991, &_var_992, &_var_993, &_var_994, &_var_995, &_var_996, &_var_997, &_var_998, &_var_999, &_var_1000, &_var_1001, &_var_1002, &_var_1003);
      _var_1616 = 15 - (uint64_t) p;
      _var_1617 = &_var_987;
      _var_1618 = &_var_988;
      _var_1619 = &_var_989;
      _var_1620 = &_var_990;
      _var_1621 = &_var_991;
      _var_1622 = &_var_992;
      _var_1623 = &_var_993;
      _var_1624 = &_var_994;
      _var_1625 = &_var_995;
      _var_1626 = &_var_996;
      _var_1627 = &_var_997;
      _var_1628 = &_var_998;
      _var_1629 = &_var_999;
      _var_1630 = &_var_1000;
      _var_1631 = &_var_1001;
      _var_1632 = &_var_1002;
      _var_1633 = &_var_1003;
      while ((_var_1616 & 0xFFFFFFFF)) {
        _var_1616 = (_var_1616 & 0xFFFFFFFF) + 4294967295;
        _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_978, *((generic64_t *) _var_1617), *((generic16_t *) _var_1618), *((generic64_t *) _var_1619), *((generic16_t *) _var_1620), *((generic64_t *) _var_1621), *((generic16_t *) _var_1622), *((generic64_t *) _var_1623), *((generic16_t *) _var_1624), *((generic64_t *) _var_1625), *((generic16_t *) _var_1626), *((generic64_t *) _var_1627), *((generic16_t *) _var_1628), *((generic64_t *) _var_1629), *((generic16_t *) _var_1630), *((generic64_t *) _var_1631), *((generic16_t *) _var_1632), &_var_847, &_var_848);
        _helper_fmul_ST0_FT0_wrapper(NULL, _var_978, *((generic64_t *) _var_1617), *((generic16_t *) _var_1618), *((generic64_t *) _var_1619), *((generic16_t *) _var_1620), *((generic64_t *) _var_1621), *((generic16_t *) _var_1622), *((generic64_t *) _var_1623), *((generic16_t *) _var_1624), *((generic64_t *) _var_1625), *((generic16_t *) _var_1626), *((generic64_t *) _var_1627), *((generic16_t *) _var_1628), *((generic64_t *) _var_1629), *((generic16_t *) _var_1630), *((generic64_t *) _var_1631), *((generic16_t *) _var_1632), '\000', '\000', *((generic8_t *) _var_1633), 'P', '\000', '\000', _var_847, _var_848, &_var_830, &_var_831, &_var_832, &_var_833, &_var_834, &_var_835, &_var_836, &_var_837, &_var_838, &_var_839, &_var_840, &_var_841, &_var_842, &_var_843, &_var_844, &_var_845, &_var_846);
        _var_1617 = &_var_830;
        _var_1618 = &_var_831;
        _var_1619 = &_var_832;
        _var_1620 = &_var_833;
        _var_1621 = &_var_834;
        _var_1622 = &_var_835;
        _var_1623 = &_var_836;
        _var_1624 = &_var_837;
        _var_1625 = &_var_838;
        _var_1626 = &_var_839;
        _var_1627 = &_var_840;
        _var_1628 = &_var_841;
        _var_1629 = &_var_842;
        _var_1630 = &_var_843;
        _var_1631 = &_var_844;
        _var_1632 = &_var_845;
        _var_1633 = &_var_846;
      }
      _helper_fmov_STN_ST0_wrapper(NULL, 1, _var_978, *((generic64_t *) _var_1617), *((generic16_t *) _var_1618), *((generic64_t *) _var_1619), *((generic16_t *) _var_1620), *((generic64_t *) _var_1621), *((generic16_t *) _var_1622), *((generic64_t *) _var_1623), *((generic16_t *) _var_1624), *((generic64_t *) _var_1625), *((generic16_t *) _var_1626), *((generic64_t *) _var_1627), *((generic16_t *) _var_1628), *((generic64_t *) _var_1629), *((generic16_t *) _var_1630), *((generic64_t *) _var_1631), *((generic16_t *) _var_1632), &_var_858, &_var_859, &_var_860, &_var_861, &_var_862, &_var_863, &_var_864, &_var_865, &_var_866, &_var_867, &_var_868, &_var_869, &_var_870, &_var_871, &_var_872, &_var_873);
      _helper_fpop_wrapper(NULL, _var_978, &_var_849, &_var_850, &_var_851, &_var_852, &_var_853, &_var_854, &_var_855, &_var_856, &_var_857);
      if (*_var_1597 == '-') {
        _helper_fxchg_ST0_STN_wrapper(NULL, 1, _var_849, _var_858, _var_859, _var_860, _var_861, _var_862, _var_863, _var_864, _var_865, _var_866, _var_867, _var_868, _var_869, _var_870, _var_871, _var_872, _var_873, &_var_603, &_var_604, &_var_605, &_var_606, &_var_607, &_var_608, &_var_609, &_var_610, &_var_611, &_var_612, &_var_613, &_var_614, &_var_615, &_var_616, &_var_617, &_var_618);
        _helper_fchs_ST0_wrapper(NULL, _var_849, _var_603, _var_604, _var_605, _var_606, _var_607, _var_608, _var_609, _var_610, _var_611, _var_612, _var_613, _var_614, _var_615, _var_616, _var_617, _var_618, &_var_587, &_var_588, &_var_589, &_var_590, &_var_591, &_var_592, &_var_593, &_var_594, &_var_595, &_var_596, &_var_597, &_var_598, &_var_599, &_var_600, &_var_601, &_var_602);
        _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_849, _var_587, _var_588, _var_589, _var_590, _var_591, _var_592, _var_593, _var_594, _var_595, _var_596, _var_597, _var_598, _var_599, _var_600, _var_601, _var_602, &_var_585, &_var_586);
        _helper_fsub_ST0_FT0_wrapper(NULL, _var_849, _var_587, _var_588, _var_589, _var_590, _var_591, _var_592, _var_593, _var_594, _var_595, _var_596, _var_597, _var_598, _var_599, _var_600, _var_601, _var_602, '\000', '\000', *((generic8_t *) _var_1633), 'P', '\000', '\000', _var_585, _var_586, &_var_568, &_var_569, &_var_570, &_var_571, &_var_572, &_var_573, &_var_574, &_var_575, &_var_576, &_var_577, &_var_578, &_var_579, &_var_580, &_var_581, &_var_582, &_var_583, &_var_584);
        _helper_fadd_STN_ST0_wrapper(NULL, 1, _var_849, _var_568, _var_569, _var_570, _var_571, _var_572, _var_573, _var_574, _var_575, _var_576, _var_577, _var_578, _var_579, _var_580, _var_581, _var_582, _var_583, '\000', '\000', _var_584, 'P', '\000', '\000', &_var_551, &_var_552, &_var_553, &_var_554, &_var_555, &_var_556, &_var_557, &_var_558, &_var_559, &_var_560, &_var_561, &_var_562, &_var_563, &_var_564, &_var_565, &_var_566, &_var_567);
        _var_1615 = _var_567;
        _helper_fpop_wrapper(NULL, _var_849, &_var_542, &_var_543, &_var_544, &_var_545, &_var_546, &_var_547, &_var_548, &_var_549, &_var_550);
        _var_1598 = _var_542;
        _helper_fchs_ST0_wrapper(NULL, _var_1598, _var_551, _var_552, _var_553, _var_554, _var_555, _var_556, _var_557, _var_558, _var_559, _var_560, _var_561, _var_562, _var_563, _var_564, _var_565, _var_566, &_var_526, &_var_527, &_var_528, &_var_529, &_var_530, &_var_531, &_var_532, &_var_533, &_var_534, &_var_535, &_var_536, &_var_537, &_var_538, &_var_539, &_var_540, &_var_541);
        _var_1599 = _var_526;
        _var_1600 = _var_527;
        _var_1601 = _var_528;
        _var_1602 = _var_529;
        _var_1603 = _var_530;
        _var_1604 = _var_531;
        _var_1605 = _var_532;
        _var_1606 = _var_533;
        _var_1607 = _var_534;
        _var_1608 = _var_535;
        _var_1609 = _var_536;
        _var_1610 = _var_537;
        _var_1611 = _var_538;
        _var_1612 = _var_539;
        _var_1613 = _var_540;
        _var_1614 = _var_541;
      } else {
        _helper_fadd_STN_ST0_wrapper(NULL, 1, _var_849, _var_858, _var_859, _var_860, _var_861, _var_862, _var_863, _var_864, _var_865, _var_866, _var_867, _var_868, _var_869, _var_870, _var_871, _var_872, _var_873, '\000', '\000', *((generic8_t *) _var_1633), 'P', '\000', '\000', &_var_509, &_var_510, &_var_511, &_var_512, &_var_513, &_var_514, &_var_515, &_var_516, &_var_517, &_var_518, &_var_519, &_var_520, &_var_521, &_var_522, &_var_523, &_var_524, &_var_525);
        _helper_fsub_STN_ST0_wrapper(NULL, 1, _var_849, _var_509, _var_510, _var_511, _var_512, _var_513, _var_514, _var_515, _var_516, _var_517, _var_518, _var_519, _var_520, _var_521, _var_522, _var_523, _var_524, '\000', '\000', _var_525, 'P', '\000', '\000', &_var_492, &_var_493, &_var_494, &_var_495, &_var_496, &_var_497, &_var_498, &_var_499, &_var_500, &_var_501, &_var_502, &_var_503, &_var_504, &_var_505, &_var_506, &_var_507, &_var_508);
        _var_1599 = _var_492;
        _var_1600 = _var_493;
        _var_1601 = _var_494;
        _var_1602 = _var_495;
        _var_1603 = _var_496;
        _var_1604 = _var_497;
        _var_1605 = _var_498;
        _var_1606 = _var_499;
        _var_1607 = _var_500;
        _var_1608 = _var_501;
        _var_1609 = _var_502;
        _var_1610 = _var_503;
        _var_1611 = _var_504;
        _var_1612 = _var_505;
        _var_1613 = _var_506;
        _var_1614 = _var_507;
        _var_1615 = _var_508;
        _helper_fpop_wrapper(NULL, _var_849, &_var_483, &_var_484, &_var_485, &_var_486, &_var_487, &_var_488, &_var_489, &_var_490, &_var_491);
        _var_1598 = _var_483;
      }
    }
    int8_t *_var_1634;
    generic64_t _var_1635;
    generic64_t _var_1636;
    _helper_fstt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 32, _var_1598, _var_1599, _var_1600, _var_1601, _var_1602, _var_1603, _var_1604, _var_1605, _var_1606, _var_1607, _var_1608, _var_1609, _var_1610, _var_1611, _var_1612, _var_1613, _var_1614);
    _helper_fpop_wrapper(NULL, _var_1598, &_var_949, &_var_950, &_var_951, &_var_952, &_var_953, &_var_954, &_var_955, &_var_956, &_var_957);
    _var_1635 = (int32_t) _stack.offset_88 > -1 ? 0 : 4294967295;
    _var_1634 = fmt_u((int64_t) ((number64_t) ((_var_1635 ^ _stack.offset_88) - _var_1635) << 32) >> 32, (int8_t *) &_stack.offset_107.member_0.member_0.member_1);
    _var_1636 = _var_1634;
    _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 32, _var_949, &_var_924, &_var_925, &_var_926, &_var_927, &_var_928, &_var_929, &_var_930, &_var_931, &_var_932, &_var_933, &_var_934, &_var_935, &_var_936, &_var_937, &_var_938, &_var_939, &_var_940, &_var_941, &_var_942, &_var_943, &_var_944, &_var_945, &_var_946, &_var_947, &_var_948);
    _helper_fpush_wrapper(NULL, _var_924, &_var_915, &_var_916, &_var_917, &_var_918, &_var_919, &_var_920, &_var_921, &_var_922, &_var_923);
    _helper_fldz_ST0_wrapper(NULL, _var_915, &_var_899, &_var_900, &_var_901, &_var_902, &_var_903, &_var_904, &_var_905, &_var_906, &_var_907, &_var_908, &_var_909, &_var_910, &_var_911, &_var_912, &_var_913, &_var_914);
    _helper_fldt_ST0_wrapper(NULL, "", _var_915, &_var_874, &_var_875, &_var_876, &_var_877, &_var_878, &_var_879, &_var_880, &_var_881, &_var_882, &_var_883, &_var_884, &_var_885, &_var_886, &_var_887, &_var_888, &_var_889, &_var_890, &_var_891, &_var_892, &_var_893, &_var_894, &_var_895, &_var_896, &_var_897, &_var_898);
    if (_var_1636 == (pointer_or_number64_t) &_stack.offset_107) {
      _stack.offset_106 = '0';
      _var_1636 = &_stack.offset_106;
    }
    generic8_t _var_1637;
    generic32_t _var_1638;
    generic32_t _var_1639;
    generic64_t _var_1640;
    union_677 *_var_1641;
    generic32_t _var_1642;
    generic16_t _var_1643;
    generic64_t _var_1644;
    generic16_t _var_1645;
    generic64_t _var_1646;
    generic16_t _var_1647;
    generic64_t _var_1648;
    generic16_t _var_1649;
    generic64_t _var_1650;
    generic16_t _var_1651;
    generic64_t _var_1652;
    generic16_t _var_1653;
    generic64_t _var_1654;
    generic16_t _var_1655;
    generic64_t _var_1656;
    generic16_t _var_1657;
    generic64_t _var_1658;
    generic16_t _var_1659;
    generic8_t _var_1660;
    _helper_fxchg_ST0_STN_wrapper(NULL, 2, _var_874, _var_883, _var_884, _var_885, _var_886, _var_887, _var_888, _var_889, _var_890, _var_891, _var_892, _var_893, _var_894, _var_895, _var_896, _var_897, _var_898, &_var_814, &_var_815, &_var_816, &_var_817, &_var_818, &_var_819, &_var_820, &_var_821, &_var_822, &_var_823, &_var_824, &_var_825, &_var_826, &_var_827, &_var_828, &_var_829);
    _var_1644 = _var_814;
    _var_1645 = _var_815;
    _var_1646 = _var_816;
    _var_1647 = _var_817;
    _var_1648 = _var_818;
    _var_1649 = _var_819;
    _var_1650 = _var_820;
    _var_1651 = _var_821;
    _var_1652 = _var_822;
    _var_1653 = _var_823;
    _var_1654 = _var_824;
    _var_1655 = _var_825;
    _var_1656 = _var_826;
    _var_1657 = _var_827;
    _var_1658 = _var_828;
    _var_1659 = _var_829;
    _var_1640 = (_var_1378 & 0xFFFFFFFFFFFFFF00) | _stack.offset_24.member_0;
    _var_1637 = _stack.offset_24.member_0 & 0x20;
    *((generic8_t *) (_var_1636 - 1)) = ((number8_t) ((uint32_t) _stack.offset_88 >> 30) & 0x2) + '+';
    *((generic8_t *) (_var_1636 - 2)) = _stack.offset_24.member_0 + '\017';
    _var_1639 = _helper_fnstcw_wrapper(NULL, 895);
    _stack.offset_78 = (number16_t) _var_1639;
    _stack.offset_76 = (number16_t) _var_1639 | 0xC00;
    _var_1638 = !p ? 64 : 0;
    _var_1641 = &_stack.offset_107;
    _var_1642 = _var_874;
    _var_1643 = 895;
    _var_1660 = _var_1615;
    while (true) {
      generic32_t _var_1661;
      generic64_t _var_1662;
      generic64_t _var_1663;
      generic64_t _var_1664;
      generic64_t _var_1665;
      generic8_t _var_1666;
      generic64_t _var_1667;
      generic32_t _var_1668;
      generic64_t _var_1669;
      _helper_fpush_wrapper(NULL, _var_1642, &_var_689, &_var_690, &_var_691, &_var_692, &_var_693, &_var_694, &_var_695, &_var_696, &_var_697);
      _helper_fmov_ST0_STN_wrapper(NULL, 1, _var_689, _var_1644, _var_1645, _var_1646, _var_1647, _var_1648, _var_1649, _var_1650, _var_1651, _var_1652, _var_1653, _var_1654, _var_1655, _var_1656, _var_1657, _var_1658, _var_1659, &_var_673, &_var_674, &_var_675, &_var_676, &_var_677, &_var_678, &_var_679, &_var_680, &_var_681, &_var_682, &_var_683, &_var_684, &_var_685, &_var_686, &_var_687, &_var_688);
      _helper_fldcw_wrapper(NULL, (uint32_t) _stack.offset_76, _var_1643, &_var_670, &_var_671, &_var_672);
      _var_1661 = _helper_fistl_ST0_wrapper(NULL, _var_689, _var_673, _var_674, _var_675, _var_676, _var_677, _var_678, _var_679, _var_680, _var_681, _var_682, _var_683, _var_684, _var_685, _var_686, _var_687, _var_688, _var_671, _var_1660, &_var_669);
      _stack.offset_24.member_1 = _var_1661;
      _helper_fpop_wrapper(NULL, _var_689, &_var_660, &_var_661, &_var_662, &_var_663, &_var_664, &_var_665, &_var_666, &_var_667, &_var_668);
      _helper_fldcw_wrapper(NULL, (uint32_t) _stack.offset_78, _var_670, &_var_657, &_var_658, &_var_659);
      _helper_fildl_FT0_wrapper(NULL, _stack.offset_24.member_1, &_var_655, &_var_656);
      _helper_fsub_ST0_FT0_wrapper(NULL, _var_660, _var_673, _var_674, _var_675, _var_676, _var_677, _var_678, _var_679, _var_680, _var_681, _var_682, _var_683, _var_684, _var_685, _var_686, _var_687, _var_688, '\000', _var_658, _var_669, _var_659, '\000', '\000', _var_655, _var_656, &_var_638, &_var_639, &_var_640, &_var_641, &_var_642, &_var_643, &_var_644, &_var_645, &_var_646, &_var_647, &_var_648, &_var_649, &_var_650, &_var_651, &_var_652, &_var_653, &_var_654);
      _var_1664 = (pointer_or_number64_t) _var_1641 + 1;
      _var_1665 = (_var_1640 & 0xFFFFFF00) | (_var_1637 | "0123456789ABCDEF"[(int64_t) _stack.offset_24.member_1]);
      _helper_fmov_FT0_STN_wrapper(NULL, 2, _var_660, _var_638, _var_639, _var_640, _var_641, _var_642, _var_643, _var_644, _var_645, _var_646, _var_647, _var_648, _var_649, _var_650, _var_651, _var_652, _var_653, &_var_636, &_var_637);
      _helper_fmul_ST0_FT0_wrapper(NULL, _var_660, _var_638, _var_639, _var_640, _var_641, _var_642, _var_643, _var_644, _var_645, _var_646, _var_647, _var_648, _var_649, _var_650, _var_651, _var_652, _var_653, '\000', _var_658, _var_654, _var_659, '\000', '\000', _var_636, _var_637, &_var_619, &_var_620, &_var_621, &_var_622, &_var_623, &_var_624, &_var_625, &_var_626, &_var_627, &_var_628, &_var_629, &_var_630, &_var_631, &_var_632, &_var_633, &_var_634, &_var_635);
      _var_1666 = _var_635;
      *((generic8_t *) _var_1641) = _var_1637 | "0123456789ABCDEF"[(int64_t) _stack.offset_24.member_1];
      _var_1669 = (pointer_or_number64_t) _var_1641 - (number64_t) &_stack.offset_107;
      _var_1667 = &_stack.offset_107;
      _var_1668 = 17;
      if ((pointer_or_number64_t) _var_1641 == (pointer_or_number64_t) &_stack.offset_107) {
        generic64_t _var_1670;
        generic64_t _var_1671;
        generic32_t _var_1672;
        generic64_t _var_1673;
        _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_660, _var_619, _var_620, _var_621, _var_622, _var_623, _var_624, _var_625, _var_626, _var_627, _var_628, _var_629, _var_630, _var_631, _var_632, _var_633, _var_634, &_var_481, &_var_482);
        _helper_fucomi_ST0_FT0_wrapper(NULL, (pointer_or_number64_t) _var_1641 - (number64_t) &_stack.offset_107, 17, (pointer_or_number64_t) &_stack + 107, 0, _var_660, _var_619, _var_620, _var_621, _var_622, _var_623, _var_624, _var_625, _var_626, _var_627, _var_628, _var_629, _var_630, _var_631, _var_632, _var_633, _var_634, _var_635, _var_481, _var_482, &_var_479, &_var_480);
        _var_1671 = _var_479;
        _var_1670 = !(_var_1671 & 0x40) ? ((number64_t) &_stack.offset_107 & 0xFFFFFFFFFFFFFF00) | 0x1 : (_var_1640 & 0xFFFFFF00) | ((_var_1671 >> 2) & 0x1);
        _var_1673 = _var_1670 & 0xFFFFFF01;
        _var_1672 = 22;
        if (!(_var_1670 & 0x1)) {
          generic64_t _var_1674;
          generic32_t _var_1675;
          generic64_t _var_1676;
          _var_1676 = _lshift((uint64_t) p, 4294967272);
          _var_1671 = (((_llvm_ctpop_i32((number32_t) p & 0xFF) << 2) & 0x4) | _var_1638 | ((number32_t) _var_1676 & 0x80)) ^ 0x4;
          _var_1675 = ((number8_t) (_var_1671 >> 4) ^ (number8_t) (((_llvm_ctpop_i32((number32_t) p & 0xFF) << 2) & 0x4) | _var_1638 | ((number32_t) _var_1676 & 0x80))) < '@' ? 1 : 24;
          _var_1672 = _var_1675;
          _var_1674 = ((number8_t) (_var_1671 >> 4) ^ (number8_t) (((_llvm_ctpop_i32((number32_t) p & 0xFF) << 2) & 0x4) | _var_1638 | ((number32_t) _var_1676 & 0x80))) < '@' ? (uint64_t) p : *((generic32_t *) &_stack.offset_20) & 0x8;
          _var_1673 = _var_1674;
          if (!(((number8_t) (_var_1671 >> 4) ^ (number8_t) (((_llvm_ctpop_i32((number32_t) p & 0xFF) << 2) & 0x4) | _var_1638 | ((number32_t) _var_1676 & 0x80))) < '@' || (*((generic32_t *) &_stack.offset_20) & 0x8) != 0)) {
            _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_660, _var_619, _var_620, _var_621, _var_622, _var_623, _var_624, _var_625, _var_626, _var_627, _var_628, _var_629, _var_630, _var_631, _var_632, _var_633, _var_634, &_var_334, &_var_335, &_var_336, &_var_337, &_var_338, &_var_339, &_var_340, &_var_341, &_var_342, &_var_343, &_var_344, &_var_345, &_var_346, &_var_347, &_var_348, &_var_349);
            _helper_fpop_wrapper(NULL, _var_660, &_var_325, &_var_326, &_var_327, &_var_328, &_var_329, &_var_330, &_var_331, &_var_332, &_var_333);
            _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_325, _var_334, _var_335, _var_336, _var_337, _var_338, _var_339, _var_340, _var_341, _var_342, _var_343, _var_344, _var_345, _var_346, _var_347, _var_348, _var_349, &_var_309, &_var_310, &_var_311, &_var_312, &_var_313, &_var_314, &_var_315, &_var_316, &_var_317, &_var_318, &_var_319, &_var_320, &_var_321, &_var_322, &_var_323, &_var_324);
            _helper_fpop_wrapper(NULL, _var_325, &_var_300, &_var_301, &_var_302, &_var_303, &_var_304, &_var_305, &_var_306, &_var_307, &_var_308);
            _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_300, _var_309, _var_310, _var_311, _var_312, _var_313, _var_314, _var_315, _var_316, _var_317, _var_318, _var_319, _var_320, _var_321, _var_322, _var_323, _var_324, &_var_284, &_var_285, &_var_286, &_var_287, &_var_288, &_var_289, &_var_290, &_var_291, &_var_292, &_var_293, &_var_294, &_var_295, &_var_296, &_var_297, &_var_298, &_var_299);
            _helper_fpop_wrapper(NULL, _var_300, &_var_275, &_var_276, &_var_277, &_var_278, &_var_279, &_var_280, &_var_281, &_var_282, &_var_283);
            _var_1663 = (pointer_or_number64_t) _var_1641 + 1;
            if (!p) {
              _var_1662 = _var_1663 - (number64_t) &_stack.offset_107 + ((pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2));
            } else {
              _var_1662 = (pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2) + (uint64_t) p + 2;
              if ((int64_t) (_var_1663 - (number64_t) &_stack.offset_107 - 1) > p) {
                _var_1662 = _var_1663 - (number64_t) &_stack.offset_107 + ((pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2));
              }
            }
            *((generic64_t *) &_stack.offset_24) = _var_1636 - 2;
            _var_1375 = _stack.offset_48.member_0 + (number32_t) _var_1662;
            pad(f, (int8_t) 32, (int32_t) _stack.offset_16, (int32_t) _var_1375, (int32_t) *((generic32_t *) &_stack.offset_20));
            out(f, (const int8_t *) _var_1597, (size_t) _stack.offset_48.member_0);
            pad(f, (int8_t) 48, (int32_t) _stack.offset_16, (int32_t) _var_1375, (int32_t) (*((generic32_t *) &_stack.offset_20) ^ 0x10000));
            out(f, (const int8_t *) &_stack.offset_107.member_0.member_0.member_1, _var_1663 - (number64_t) &_stack.offset_107);
            pad(f, (int8_t) 48, (int32_t) (number32_t) (_var_1662 - (_var_1663 - (number64_t) &_stack.offset_107 + ((pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2)))), (int32_t) 0, (int32_t) 0);
            _var_1377 = *((generic64_t *) &_stack.offset_24);
            _var_1376 = (pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2);
            break;
          }
        }
        _var_1667 = _var_1671;
        _var_1668 = _var_1672;
        _var_1669 = _var_1673;
        *((generic8_t *) ((pointer_or_number64_t) _var_1641 + 1)) = '.';
        _var_1664 = &_stack.offset_107.member_0.member_0.member_0.offset_1.member_1.offset_1;
        _var_1665 = _var_1670 & 0xFFFFFF01;
        _var_1666 = _var_480;
      }
      _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_660, _var_619, _var_620, _var_621, _var_622, _var_623, _var_624, _var_625, _var_626, _var_627, _var_628, _var_629, _var_630, _var_631, _var_632, _var_633, _var_634, &_var_477, &_var_478);
      _helper_fucomi_ST0_FT0_wrapper(NULL, _var_1669, _var_1668, _var_1667, 0, _var_660, _var_619, _var_620, _var_621, _var_622, _var_623, _var_624, _var_625, _var_626, _var_627, _var_628, _var_629, _var_630, _var_631, _var_632, _var_633, _var_634, _var_1666, _var_477, _var_478, &_var_475, &_var_476);
      _var_1660 = _var_476;
      _var_1642 = _var_660;
      _var_1643 = _var_657;
      _var_1644 = _var_619;
      _var_1645 = _var_620;
      _var_1646 = _var_621;
      _var_1647 = _var_622;
      _var_1648 = _var_623;
      _var_1649 = _var_624;
      _var_1650 = _var_625;
      _var_1651 = _var_626;
      _var_1652 = _var_627;
      _var_1653 = _var_628;
      _var_1654 = _var_629;
      _var_1655 = _var_630;
      _var_1656 = _var_631;
      _var_1657 = _var_632;
      _var_1658 = _var_633;
      _var_1659 = _var_634;
      if ((_var_475 & 0x44) != 64) {
        continue;
      }
      _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_660, _var_619, _var_620, _var_621, _var_622, _var_623, _var_624, _var_625, _var_626, _var_627, _var_628, _var_629, _var_630, _var_631, _var_632, _var_633, _var_634, &_var_409, &_var_410, &_var_411, &_var_412, &_var_413, &_var_414, &_var_415, &_var_416, &_var_417, &_var_418, &_var_419, &_var_420, &_var_421, &_var_422, &_var_423, &_var_424);
      _helper_fpop_wrapper(NULL, _var_660, &_var_400, &_var_401, &_var_402, &_var_403, &_var_404, &_var_405, &_var_406, &_var_407, &_var_408);
      _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_400, _var_409, _var_410, _var_411, _var_412, _var_413, _var_414, _var_415, _var_416, _var_417, _var_418, _var_419, _var_420, _var_421, _var_422, _var_423, _var_424, &_var_384, &_var_385, &_var_386, &_var_387, &_var_388, &_var_389, &_var_390, &_var_391, &_var_392, &_var_393, &_var_394, &_var_395, &_var_396, &_var_397, &_var_398, &_var_399);
      _helper_fpop_wrapper(NULL, _var_400, &_var_375, &_var_376, &_var_377, &_var_378, &_var_379, &_var_380, &_var_381, &_var_382, &_var_383);
      _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_375, _var_384, _var_385, _var_386, _var_387, _var_388, _var_389, _var_390, _var_391, _var_392, _var_393, _var_394, _var_395, _var_396, _var_397, _var_398, _var_399, &_var_359, &_var_360, &_var_361, &_var_362, &_var_363, &_var_364, &_var_365, &_var_366, &_var_367, &_var_368, &_var_369, &_var_370, &_var_371, &_var_372, &_var_373, &_var_374);
      _helper_fpop_wrapper(NULL, _var_375, &_var_350, &_var_351, &_var_352, &_var_353, &_var_354, &_var_355, &_var_356, &_var_357, &_var_358);
      _var_1663 = _var_1664;
      if (!p) {
        _var_1662 = _var_1663 - (number64_t) &_stack.offset_107 + ((pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2));
      } else {
        _var_1662 = (pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2) + (uint64_t) p + 2;
        if ((int64_t) (_var_1663 - (number64_t) &_stack.offset_107 - 1) > p) {
          _var_1662 = _var_1663 - (number64_t) &_stack.offset_107 + ((pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2));
        }
      }
      *((generic64_t *) &_stack.offset_24) = _var_1636 - 2;
      _var_1375 = _stack.offset_48.member_0 + (number32_t) _var_1662;
      pad(f, (int8_t) 32, (int32_t) _stack.offset_16, (int32_t) _var_1375, (int32_t) *((generic32_t *) &_stack.offset_20));
      out(f, (const int8_t *) _var_1597, (size_t) _stack.offset_48.member_0);
      pad(f, (int8_t) 48, (int32_t) _stack.offset_16, (int32_t) _var_1375, (int32_t) (*((generic32_t *) &_stack.offset_20) ^ 0x10000));
      out(f, (const int8_t *) &_stack.offset_107.member_0.member_0.member_1, _var_1663 - (number64_t) &_stack.offset_107);
      pad(f, (int8_t) 48, (int32_t) (number32_t) (_var_1662 - (_var_1663 - (number64_t) &_stack.offset_107 + ((pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2)))), (int32_t) 0, (int32_t) 0);
      _var_1377 = *((generic64_t *) &_stack.offset_24);
      _var_1376 = (pointer_or_number64_t) &_stack.offset_107 - (_var_1636 - 2);
      break;
    }
  } else {
    uint8_t *_var_1677;
    uint8_t *_var_1678;
    _helper_fmov_FT0_STN_wrapper(NULL, 0, _var_1170, _var_1179, _var_1180, _var_1181, _var_1182, _var_1183, _var_1184, _var_1185, _var_1186, _var_1187, _var_1188, _var_1189, _var_1190, _var_1191, _var_1192, _var_1193, _var_1194, &_var_1159, &_var_1160);
    _helper_fucomi_ST0_FT0_wrapper(NULL, (uint64_t) (_stack.offset_24.member_1 & 0x20), 24, (int64_t) ((((_llvm_ctpop_i32((number32_t) ((uint64_t) _var_1369 - 1) & 0xFF) << 2) & 0x4) | _var_1370 | ((((number8_t) ((uint64_t) _var_1369 - 1) + '\001') ^ (number8_t) ((uint64_t) _var_1369 - 1)) & 0x10) | _var_1373 | ((number32_t) _var_1372 & 0x80) | _var_1371) ^ 0x4), 0, _var_1170, _var_1179, _var_1180, _var_1181, _var_1182, _var_1183, _var_1184, _var_1185, _var_1186, _var_1187, _var_1188, _var_1189, _var_1190, _var_1191, _var_1192, _var_1193, _var_1194, '\000', _var_1159, _var_1160, &_var_1157, &_var_1158);
    _helper_fpop_wrapper(NULL, _var_1170, &_var_1148, &_var_1149, &_var_1150, &_var_1151, &_var_1152, &_var_1153, &_var_1154, &_var_1155, &_var_1156);
    if (!((number8_t) _var_1157 & 0x4)) {
      uint8_t *_var_1679;
      _var_1679 = !((number8_t) _stack.offset_24.member_1 & 0x20) ? (generic64_t) "INF" : (generic64_t) "inf";
      _var_1678 = _var_1679;
      if (!((number8_t) _var_1157 & 0x40)) {
        _var_1677 = !((number8_t) _stack.offset_24.member_1 & 0x20) ? (generic64_t) "NAN" : (generic64_t) "nan";
        _var_1678 = _var_1677;
      }
    } else {
      _var_1677 = !((number8_t) _stack.offset_24.member_1 & 0x20) ? (generic64_t) "NAN" : (generic64_t) "nan";
      _var_1678 = _var_1677;
    }
    _var_1377 = _var_1678;
    _var_1375 = _stack.offset_48.member_0 + 3;
    pad(f, (int8_t) 32, (int32_t) _stack.offset_16, (int32_t) _var_1375, (int32_t) (*((generic32_t *) &_stack.offset_20) & 0xFFFEFFFF));
    out(f, (const int8_t *) _var_1347, (size_t) _stack.offset_48.member_0);
    _var_1376 = 3;
  }
  generic32_t _var_1680;
  out(f, (const int8_t *) _var_1377, _var_1376);
  pad(f, (int8_t) 32, (int32_t) _stack.offset_16, (int32_t) _var_1375, (int32_t) (*((generic32_t *) &_stack.offset_20) ^ 0x2000));
  _var_1680 = (int64_t) ((number64_t) _var_1375 << 32) < (int64_t) ((number64_t) _stack.offset_16 << 32) ? _stack.offset_16 : _var_1375;
  _var_1374 = _var_1680;
  return (int32_t) _var_1374;
}

_ABI(SystemV_x86_64)
int32_t printf_core(FILE_ *f, const int8_t *fmt, va_list *ap, arg *nl_arg, int32_t *nl_type) {
  struct _PACKED struct_571 {
    uint8_t padding_at_0[4];
    generic32_t offset_4;
    generic32_t offset_8;
    uint8_t padding_at_12[4];
    union_666 offset_16;
    struct_695 *offset_24;
    union_667 offset_32;
    union_668 offset_40;
    generic32_t offset_48;
    generic32_t offset_52;
    union_605 *offset_56;
    uint8_t padding_at_64[16];
    struct_665 offset_80;
    uint8_t padding_at_96[104];
  } _stack;
  uint64_t _loop_state_var;
  const int8_t *_var_0;
  *((va_list **) &_stack.offset_16) = ap;
  *((arg **) &_stack.offset_40) = nl_arg;
  _stack.offset_24 = nl_type;
  _stack.offset_8 = 0;
  _stack.offset_4 = 0;
  _stack.offset_48 = 0;
  _var_0 = fmt;
  generic32_t _var_1;
  generic64_t _var_2;
  struct_695 *_var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  while (true) {
    generic64_t _var_7;
    generic64_t _var_8;
    generic64_t _var_9;
    _var_7 = _var_8;
    if ((int32_t) _stack.offset_4 > -1) {
      generic64_t _var_10;
      _var_10 = _stack.offset_8;
      _var_7 = 2147483647 - _stack.offset_4;
      _stack.offset_4 = _stack.offset_4 + _stack.offset_8;
      if ((int64_t) ((number64_t) _var_7 << 32) < (int64_t) ((number64_t) _var_10 << 32)) {
        int32_t *_var_11;
        _var_11 = unreserved___errno_location();
        _var_7 = _var_11;
        _stack.offset_4 = 4294967295;
        *((generic32_t *) _var_7) = 75;
      }
    }
    if (!*_var_0) {
      if (f) {
        break;
      }
      _var_1 = 0;
      if (_stack.offset_48) {
        _var_3 = _stack.offset_24;
        _var_2 = 1;
        if (!_var_3->offset_4) {
          _loop_state_var = 3;
          break;
        }
        _var_4 = 0;
        _var_5 = _var_3->offset_4;
        _var_6 = 1;
        _loop_state_var = 0;
        break;
      }
    } else {
      generic64_t _var_12;
      generic64_t _var_13;
      generic64_t _var_14;
      _var_12 = 0;
      _var_13 = _var_0;
      _var_14 = _var_7;
      generic64_t _var_15;
      generic64_t _var_16;
      while (true) {
        bool _break_from_loop_17 = false;
        _var_16 = _var_12;
        _var_15 = _var_13;
        _var_14 = (_var_14 & 0xFFFFFFFFFFFFFF00) | *((generic8_t *) _var_15);
        _var_12 = _var_16 + 1;
        _var_13 = &((const int8_t *) _var_15)[1];
        switch ((number8_t) *((generic8_t *) _var_15)) {
          case 0:
          case 37:
          {
            _break_from_loop_17 = true;
            break;
          } break;
          default:
          {
            continue;
          } break;
        }
        if (_break_from_loop_17)
          break;
      }
      generic64_t _var_18;
      generic64_t _var_19;
      _var_18 = _var_15;
      _var_19 = _var_15;
      if (*((generic8_t *) _var_15) == '%') {
        generic64_t _var_20;
        generic64_t _var_21;
        generic64_t _var_22;
        _var_20 = 0;
        _var_21 = _var_15;
        _var_22 = _var_15;
        generic64_t _var_23;
        generic64_t _var_24;
        while (true) {
          generic64_t _var_25;
          _var_25 = _var_20;
          _var_24 = _var_21;
          _var_23 = _var_22;
          if ((pointer_or_number8_t) _var_0[_var_16 + (2 * _var_25 + 1)] == '%') {
            _var_20 = _var_25 + 1;
            _var_21 = &((const int8_t *) _var_21)[2];
            _var_22 = &((const int8_t *) _var_22)[1];
            _var_23 = &_var_0[_var_25 + (_var_16 + 1)];
            _var_24 = &_var_0[_var_16 + (2 * _var_25 + 2)];
            if ((pointer_or_number8_t) _var_0[_var_16 + (2 * _var_25 + 2)] == '%') {
              continue;
            }
          }
          break;
        }
        _var_18 = _var_23;
        _var_19 = _var_24;
      }
      generic64_t _var_26;
      _var_26 = _var_18 - (number64_t) _var_0;
      _stack.offset_8 = (number32_t) _var_26;
      if (f) {
        *((generic64_t *) &_stack.offset_32) = _var_18 - (number64_t) _var_0;
        out(f, _var_0, (int64_t) ((number64_t) (_var_18 - (number64_t) _var_0) << 32) >> 32);
        _var_26 = *((generic64_t *) &_stack.offset_32);
      }
      generic64_t _var_27;
      generic64_t _var_28;
      generic64_t _var_29;
      _var_27 = _var_14;
      _var_28 = _var_19;
      _var_29 = _var_9;
      if ((_var_26 & 0xFFFFFFFF)) {
        continue;
      }
      generic64_t _var_30;
      const int8_t *_var_31;
      _var_31 = &((const int8_t *) _var_19)[1];
      _var_30 = 4294967295;
      if (!((((pointer_or_number64_t) (int64_t) ((const int8_t *) _var_19)[1] + 4294967248) & 0xFFFFFFFF) > 9)) {
        _var_30 = 4294967295;
        _var_31 = &((const int8_t *) _var_19)[1];
        if ((pointer_or_number8_t) ((const int8_t *) _var_19)[2] == '$') {
          _stack.offset_48 = 1;
          _var_31 = &((const int8_t *) _var_19)[3];
          _var_30 = ((pointer_or_number64_t) (int64_t) ((const int8_t *) _var_19)[1] + 4294967248) & 0xFFFFFFFF;
        }
      }
      generic64_t _var_32;
      generic64_t _var_33;
      generic8_t _var_34;
      generic32_t _var_35;
      _var_33 = _var_31;
      _var_34 = *((generic8_t *) _var_33);
      _var_35 = _var_34;
      _var_32 = 0;
      if (!(_var_35 > 63 || _var_35 < 32)) {
        generic64_t _var_36;
        generic32_t _var_37;
        generic32_t _var_38;
        generic8_t _var_39;
        generic64_t _var_40;
        generic64_t _var_41;
        _var_36 = 0;
        _var_37 = _var_35 - 32;
        _var_38 = _var_34;
        _var_39 = *((generic8_t *) _var_33);
        _var_40 = _var_31;
        _var_41 = 0;
        generic64_t _var_42;
        generic64_t _var_43;
        generic8_t _var_44;
        generic32_t _var_45;
        while (true) {
          generic64_t _var_46;
          generic64_t _var_47;
          _var_45 = _var_38;
          _var_44 = _var_39;
          _var_43 = _var_40;
          _var_42 = _var_41;
          _var_47 = _lshift((uint64_t) (_var_45 - 63), 4294967272);
          _var_46 = _lshift((uint64_t) ((_var_37 ^ 0x1F) & (_var_37 ^ (_var_45 - 63))), 4294967276);
          if (((0x12889 >> (_var_44 & 0x1F)) & 0x1)) {
            _var_43 = (pointer_or_number64_t) _var_31 + 1 + _var_36;
            _var_40 = _var_40 + 1;
            _var_41 = _var_41 | (0x1 << (_var_44 & 0x1F));
            _var_42 = _var_41;
            _var_39 = *((generic8_t *) _var_43);
            _var_44 = _var_39;
            _var_38 = _var_44;
            _var_45 = _var_38;
            _var_37 = _var_45 - 32;
            _var_36 = _var_36 + 1;
            if (!(_var_45 > 63 || _var_45 < 32)) {
              continue;
            }
          }
          break;
        }
        _var_32 = _var_42;
        _var_33 = _var_43;
        _var_34 = _var_44;
        _var_35 = _var_45;
      }
      generic32_t _var_48;
      generic64_t _var_49;
      generic64_t _var_50;
      generic64_t _var_51;
      generic64_t _var_52;
      if (_var_34 == '*') {
        generic64_t _var_53;
        generic64_t _var_54;
        generic64_t _var_55;
        generic64_t _var_56;
        generic64_t _var_57;
        generic64_t _var_58;
        if ((!((((pointer_or_number64_t) *((generic8_t *) (_var_33 + 1)) + 4294967248) & 0xFFFFFFFE) > 9)) && (*((generic8_t *) (_var_33 + 2)) == '$')) {
          _var_48 = 4294967295;
          if (_stack.offset_48) {
            _var_1 = _var_48;
            break;
          }
          _var_55 = _var_35;
          _var_53 = 0;
          _var_54 = _var_33 + 1;
          if (f) {
            if (*((generic32_t *) *((generic64_t *) &_stack.offset_16)) > 47) {
              _var_56 = *((generic64_t *) (*((generic64_t *) &_stack.offset_16) + 8));
              *((generic64_t *) (*((generic64_t *) &_stack.offset_16) + 8)) = _var_56 + 8;
              _var_57 = _var_33 + 1;
              _var_58 = *((generic64_t *) &_stack.offset_16);
            } else {
              _var_56 = *((generic64_t *) (*((generic64_t *) &_stack.offset_16) + 16)) + *((generic32_t *) *((generic64_t *) &_stack.offset_16));
              *((generic32_t *) *((generic64_t *) &_stack.offset_16)) = *((generic32_t *) *((generic64_t *) &_stack.offset_16)) + 8;
              _var_57 = _var_33 + 1;
              _var_58 = _var_35;
            }
            _var_54 = _var_57;
            _var_55 = _var_58;
            _var_53 = *((generic32_t *) _var_56);
          }
        } else {
          _var_58 = _stack.offset_24;
          _stack.offset_48 = 1;
          _var_57 = _var_33 + 3;
          *((generic32_t *) (((number64_t) *((generic8_t *) (_var_33 + 1)) << 2) + _var_58 - 192)) = 10;
          _var_56 = ((number64_t) *((generic8_t *) (_var_33 + 1)) << 4) + *((generic64_t *) &_stack.offset_40) - 768;
          _var_54 = _var_57;
          _var_55 = _var_58;
          _var_53 = *((generic32_t *) _var_56);
        }
        _var_50 = _var_53;
        _var_51 = _var_54;
        _var_52 = _var_55;
        _var_49 = _var_32;
        if (!(_var_50 < 2147483648)) {
          _var_49 = (_var_32 & 0xFFFFDFFF) | 0x2000;
          _var_50 = (0 - _var_53) & 0xFFFFFFFF;
          _var_51 = _var_54;
          _var_52 = _var_55;
        }
      } else {
        generic64_t _var_59;
        generic64_t _var_60;
        _var_59 = _var_26;
        _var_60 = _var_33;
        if (!((((pointer_or_number64_t) *((generic8_t *) _var_33) + 4294967248) & 0xFFFFFFFF) > 9)) {
          generic64_t _var_61;
          generic64_t _var_62;
          generic64_t _var_63;
          _var_61 = 0;
          _var_62 = ((pointer_or_number64_t) *((generic8_t *) _var_33) + 4294967248) & 0xFFFFFFFF;
          _var_63 = _var_26;
          generic64_t _var_64;
          do {
            _var_64 = _var_33 + 1 + _var_61;
            _var_63 = ((_var_63 * 10) & 0xFFFFFFFE) + _var_62;
            _var_62 = ((pointer_or_number64_t) *((generic8_t *) _var_64) + 4294967248) & 0xFFFFFFFF;
            _var_61 = _var_61 + 1;
          } while (!(_var_62 > 9));
          _var_59 = _var_63;
          _var_60 = _var_64;
        }
        _var_51 = _var_60;
        _var_52 = _var_35;
        _var_50 = _var_59 & 0xFFFFFFFF;
        _var_48 = 4294967295;
        _var_49 = _var_32;
        if ((_var_59 & 0x80000000)) {
          _var_1 = _var_48;
          break;
        }
      }
      generic32_t _var_65;
      generic64_t _var_66;
      generic64_t _var_67;
      _var_66 = _var_51;
      _var_67 = _var_52;
      _var_65 = 4294967295;
      if (*((generic8_t *) _var_66) == '.') {
        if (*((generic8_t *) (_var_51 + 1)) == '*') {
          if ((!((((pointer_or_number64_t) *((generic8_t *) (_var_51 + 2)) + 4294967248) & 0xFFFFFFFE) > 9)) && (*((generic8_t *) (_var_51 + 3)) == '$')) {
            _var_48 = 4294967295;
            if (_stack.offset_48) {
              _var_1 = _var_48;
              break;
            }
            _var_65 = 0;
            _var_66 = _var_51 + 2;
            _var_67 = _var_52;
            if (f) {
              generic64_t _var_68;
              generic64_t _var_69;
              if (*((generic32_t *) *((generic64_t *) &_stack.offset_16)) > 47) {
                _var_69 = *((generic64_t *) (*((generic64_t *) &_stack.offset_16) + 8));
                *((generic64_t *) (*((generic64_t *) &_stack.offset_16) + 8)) = _var_69 + 8;
                _var_68 = *((generic64_t *) &_stack.offset_16);
              } else {
                _var_69 = *((generic64_t *) (*((generic64_t *) &_stack.offset_16) + 16)) + *((generic32_t *) *((generic64_t *) &_stack.offset_16));
                *((generic32_t *) *((generic64_t *) &_stack.offset_16)) = *((generic32_t *) *((generic64_t *) &_stack.offset_16)) + 8;
                _var_68 = _var_52;
              }
              _var_67 = _var_68;
              _var_65 = *((generic32_t *) _var_69);
              _var_66 = _var_51 + 2;
            }
          } else {
            _var_67 = _stack.offset_24;
            _var_66 = _var_51 + 4;
            *((generic32_t *) (((number64_t) *((generic8_t *) (_var_51 + 2)) << 2) + _var_67 - 192)) = 10;
            _var_65 = *((generic32_t *) (((number64_t) *((generic8_t *) (_var_51 + 2)) << 4) + *((generic64_t *) &_stack.offset_40) - 768));
          }
        } else {
          generic64_t _var_70;
          generic64_t _var_71;
          _var_70 = _var_26;
          _var_71 = _var_51 + 1;
          if (!((((pointer_or_number64_t) *((generic8_t *) (_var_51 + 1)) + 4294967248) & 0xFFFFFFFF) > 9)) {
            generic64_t _var_72;
            generic64_t _var_73;
            generic64_t _var_74;
            _var_72 = 0;
            _var_73 = ((pointer_or_number64_t) *((generic8_t *) (_var_51 + 1)) + 4294967248) & 0xFFFFFFFF;
            _var_74 = _var_26;
            generic64_t _var_75;
            do {
              _var_75 = _var_51 + 2 + _var_72;
              _var_74 = ((_var_74 * 10) & 0xFFFFFFFE) + _var_73;
              _var_73 = ((pointer_or_number64_t) *((generic8_t *) _var_75) + 4294967248) & 0xFFFFFFFF;
              _var_72 = _var_72 + 1;
            } while (!(_var_73 > 9));
            _var_70 = _var_74;
            _var_71 = _var_75;
          }
          _var_66 = _var_71;
          _var_65 = (number32_t) _var_70;
          _var_67 = _var_52;
        }
      }
      generic64_t _var_76;
      generic64_t _var_77;
      generic64_t _var_78;
      generic64_t _var_79;
      _var_78 = _var_66;
      _var_76 = _var_78 + 1;
      _var_77 = 0;
      _var_79 = 0;
      generic64_t _var_80;
      generic64_t _var_81;
      generic64_t _var_82;
      while (true) {
        _var_80 = _var_79;
        _var_81 = _var_78;
        if (!((uint32_t) *((generic8_t *) _var_81) > 122 || (uint32_t) *((generic8_t *) _var_81) < 65)) {
          _var_82 = _var_76 + _var_77;
          _var_78 = _var_78 + 1;
          _var_79 = *((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.states + _var_80 * 58 + (number64_t) ((pointer_or_number32_t) *((generic8_t *) _var_81) - 65) * 1));
          _var_77 = _var_77 + 1;
          if (*((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.states + _var_80 * 58 + (number64_t) ((pointer_or_number32_t) *((generic8_t *) _var_81) - 65) * 1)) < 9 && *((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.states + _var_80 * 58 + (number64_t) ((pointer_or_number32_t) *((generic8_t *) _var_81) - 65) * 1)) > 0) {
            continue;
          }
          break;
        }
        _var_1 = 4294967295;
        _loop_state_var = 1;
        break;
      }
      if (_loop_state_var == 1) {
        break;
      }
      _var_48 = 4294967295;
      if (*((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.states + _var_80 * 58 + (number64_t) ((pointer_or_number32_t) *((generic8_t *) _var_81) - 65) * 1))) {
        generic64_t _var_83;
        generic64_t _var_84;
        generic64_t _var_85;
        generic64_t _var_86;
        generic64_t _var_87;
        generic64_t _var_88;
        if (*((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.states + _var_80 * 58 + (number64_t) ((pointer_or_number32_t) *((generic8_t *) _var_81) - 65) * 1)) == '\025') {
          _var_48 = 4294967295;
          _var_86 = _var_30;
          _var_87 = _var_67;
          _var_88 = _var_9;
          if (_var_30 < 2147483648) {
            _var_1 = _var_48;
            break;
          }
          _var_27 = _var_86;
          _var_84 = _var_87;
          _var_85 = _var_88;
          _var_29 = _var_85;
          _var_28 = _var_82;
          _var_83 = _var_26;
          if (!f) {
            continue;
          }
        } else {
          if (_var_30 < 2147483648) {
            _var_87 = _stack.offset_24;
            *((generic32_t *) (((int64_t) ((number64_t) _var_30 << 32) >> 30) + _var_87)) = *((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.states + _var_80 * 58 + (number64_t) ((pointer_or_number32_t) *((generic8_t *) _var_81) - 65) * 1));
            _var_86 = (int64_t) ((number64_t) _var_30 << 32) >> 28;
            _var_88 = *((generic64_t *) (_var_86 + *((generic64_t *) &_stack.offset_40)));
            _stack.offset_80.offset_0 = _var_88;
            _stack.offset_80.offset_8 = *((generic64_t *) (_var_86 + *((generic64_t *) &_stack.offset_40) + 8));
            _var_27 = _var_86;
            _var_84 = _var_87;
            _var_85 = _var_88;
            _var_29 = _var_85;
            _var_28 = _var_82;
            _var_83 = _var_26;
            if (!f) {
              continue;
            }
          } else {
            _var_48 = 0;
            if (!f) {
              _var_1 = _var_48;
              break;
            }
            *((generic64_t *) &_stack.offset_32) = _var_26;
            pop_arg((arg *) &_stack.offset_80, (int32_t) *((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.states + _var_80 * 58 + (number64_t) ((pointer_or_number32_t) *((generic8_t *) _var_81) - 65) * 1)), (va_list *) *((generic64_t *) &_stack.offset_16));
            _var_83 = *((generic64_t *) &_stack.offset_32);
            _var_84 = &_stack.offset_80;
            _var_85 = _var_9;
          }
        }
        generic64_t _var_89;
        generic64_t _var_90;
        generic64_t _var_91;
        generic64_t _var_92;
        _var_92 = (uint64_t) *((generic8_t *) _var_81);
        _var_91 = 0;
        if (!_var_80) {
          _var_90 = _var_92;
          _var_89 = _lshift(_var_91, 4294967272);
        } else {
          if (!(number8_t) ((*((generic8_t *) _var_81) & 0xF) - 3)) {
            _var_92 = (number32_t) *((generic8_t *) _var_81) & 0xFFFFFFDF;
            _var_91 = _var_92;
            _var_90 = _var_92;
            _var_89 = _lshift(_var_91, 4294967272);
          } else {
            generic64_t _var_93;
            generic64_t _var_94;
            _var_94 = _lshift(((*((generic8_t *) _var_81) & 0xF) - 3) & 0xFF, 0);
            _var_93 = _lshift((((*((generic8_t *) _var_81) & 0xF) - 3) ^ ((number8_t) ((*((generic8_t *) _var_81) & 0xF) - 3) + '\003')) & (((number8_t) ((*((generic8_t *) _var_81) & 0xF) - 3) + '\003') ^ 0x3), 4);
            _var_90 = (uint64_t) *((generic8_t *) _var_81);
          }
        }
        generic64_t _var_95;
        generic64_t _var_96;
        generic64_t _var_97;
        generic64_t _var_98;
        generic64_t _var_99;
        generic64_t _var_100;
        generic64_t _var_101;
        generic64_t _var_102;
        _var_97 = !(_var_49 & 0x2000) ? _var_49 : _var_49 & 0xFFFEFFFF;
        _var_99 = _var_97;
        _var_100 = "-+   0X0x";
        _var_98 = _var_65;
        _var_101 = (pointer_or_number64_t) &_stack.offset_80 + 64;
        _var_102 = _var_0;
        if (((_var_90 + 4294967231) & 0xFFFFFFFF) > 55) {
          *((generic64_t *) &_stack.offset_32) = _var_101 - _var_102;
          _var_27 = _stack.offset_8;
          _var_96 = (int64_t) ((int64_t) ((number64_t) _var_98 << 32) >> 32) < (int64_t) (_var_101 - _var_102) ? _var_101 - _var_102 : _var_98;
          _stack.offset_56 = _var_100;
          _stack.offset_52 = (number32_t) ((_var_96 & 0xFFFFFFFF) + _var_27);
          _var_95 = (int64_t) ((number64_t) ((_var_96 & 0xFFFFFFFF) + _var_27) << 32) < (int64_t) ((number64_t) _var_50 << 32) ? _var_50 : (_var_96 & 0xFFFFFFFF) + _var_27;
          pad(f, (int8_t) 32, (int32_t) (number32_t) _var_95, (int32_t) (number32_t) ((_var_96 & 0xFFFFFFFF) + _var_27), (int32_t) (number32_t) _var_99);
          out(f, (const int8_t *) &_stack.offset_56->member_0.member_1, (size_t) _stack.offset_8);
          _stack.offset_8 = _stack.offset_52;
          pad(f, (int8_t) 48, (int32_t) (number32_t) _var_95, (int32_t) _stack.offset_52, (int32_t) ((number32_t) _var_99 ^ 0x10000));
          pad(f, (int8_t) 48, (int32_t) (number32_t) _var_96, (int32_t) _stack.offset_32.member_1, (int32_t) 0);
          out(f, (const int8_t *) _var_102, *((generic64_t *) &_stack.offset_32));
          pad(f, (int8_t) 32, (int32_t) (number32_t) _var_95, (int32_t) _stack.offset_8, (int32_t) ((number32_t) _var_99 ^ 0x2000));
          _stack.offset_8 = (number32_t) _var_95;
          _var_28 = _var_82;
          _var_29 = _var_85;
          continue;
        }
        generic64_t _var_103;
        generic64_t _var_104;
        generic64_t _var_105;
        generic32_t _var_106;
        generic64_t _var_107;
        bool _break_from_loop_108 = false;
        _var_100 = "-+   0X0x";
        _var_98 = _var_65;
        _var_99 = _var_97;
        _var_101 = (pointer_or_number64_t) &_stack.offset_80 + 64;
        _var_102 = _var_0;
        _var_103 = _var_65;
        _var_104 = _var_97;
        _var_105 = _var_90;
        _var_106 = _var_65;
        _var_107 = (_var_90 + 4294967231) & 0xFFFFFFFF;
        switch ((number64_t) *((generic64_t *) &"\t1@"[8 * ((_var_90 + 4294967231) & 0xFFFFFFFF)])) {
          case 4206750:
          case 4206781:
          {
            if ((number64_t) *((generic64_t *) &"\t1@"[8 * ((_var_90 + 4294967231) & 0xFFFFFFFF)]) == 4206750) {
              *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_56 + 20)) = 0;
              *((generic32_t *) ((pointer_or_number64_t) &_stack.offset_56 + 16)) = (number32_t) _stack.offset_80.offset_0;
              _stack.offset_80.offset_0 = (pointer_or_number64_t) &_stack.offset_56 + 16;
              _var_107 = (pointer_or_number64_t) &_stack.offset_56 + 16;
              _var_106 = 4294967295;
            }
            generic64_t _var_109;
            generic64_t _var_110;
            _var_110 = _var_107;
            _var_109 = _var_83;
            if ((number32_t) _var_83 - _var_106 > ~_var_106) {
              _var_109 = _var_83;
              _var_110 = _var_107;
              if (*((generic32_t *) _stack.offset_80.offset_0)) {
                generic32_t _var_111;
                generic64_t _var_112;
                generic64_t _var_113;
                generic32_t _var_114;
                _var_111 = *((generic32_t *) _stack.offset_80.offset_0);
                _var_112 = _stack.offset_80.offset_0;
                _var_113 = _var_83;
                _var_114 = (number32_t) _var_83;
                int32_t _var_115;
                generic64_t _var_116;
                while (true) {
                  *((generic64_t *) &_stack.offset_32) = _var_112 + 4;
                  _var_115 = wctomb((int8_t *) ((pointer_or_number64_t) &_stack.offset_56 + 12), (wchar_t) _var_111);
                  if (_var_115 > -1) {
                    _var_116 = _var_113;
                    if (!(_var_106 - (_var_114 + (pointer_or_number32_t) _var_115) > ~(generic32_t) _var_115)) {
                      generic32_t _var_117;
                      _var_116 = (_var_113 & 0xFFFFFFFF) + (uint64_t) _var_115;
                      _var_117 = (number32_t) _var_116;
                      if (_var_117 - _var_106 > ~_var_106) {
                        _var_112 = *((generic64_t *) &_stack.offset_32);
                        _var_111 = *((generic32_t *) _var_112);
                        _var_116 = (_var_113 & 0xFFFFFFFF) + (uint64_t) _var_115;
                        _var_113 = (_var_113 & 0xFFFFFFFF) + (uint64_t) _var_115;
                        _var_114 = _var_117;
                        if (_var_111) {
                          continue;
                        }
                      }
                    }
                    break;
                  }
                  _var_1 = 4294967295;
                  _loop_state_var = 1;
                  break;
                }
                if (_loop_state_var == 1) {
                  _break_from_loop_108 = true;
                  break;
                }
                _var_109 = _var_116;
                _var_110 = (uint64_t) _var_115;
              }
            }
            generic64_t _var_118;
            _var_118 = _var_110;
            pad(f, (int8_t) 32, (int32_t) (number32_t) _var_50, (int32_t) (number32_t) _var_109, (int32_t) (number32_t) _var_97);
            if ((_var_109 & 0xFFFFFFFF) > _stack.offset_8) {
              _var_118 = _var_110;
              if (*((generic32_t *) _stack.offset_80.offset_0)) {
                generic64_t _var_119;
                generic64_t _var_120;
                generic32_t _var_121;
                _var_119 = _stack.offset_80.offset_0 + 4;
                _var_120 = 0;
                _var_121 = *((generic32_t *) _stack.offset_80.offset_0);
                int32_t _var_122;
                while (true) {
                  _var_122 = wctomb((int8_t *) ((pointer_or_number64_t) &_stack.offset_56 + 12), (wchar_t) _var_121);
                  _stack.offset_8 = _stack.offset_8 + (pointer_or_number32_t) _var_122;
                  if (!((int64_t) ((int64_t) ((number64_t) _var_109 << 32) >> 32) < _stack.offset_8)) {
                    out(f, (const int8_t *) ((pointer_or_number64_t) &_stack.offset_56 + 12), (size_t) _var_122);
                    if ((_var_109 & 0xFFFFFFFF) > _stack.offset_8) {
                      _var_121 = *((generic32_t *) (_var_119 + (_var_120 << 2)));
                      _var_120 = _var_120 + 1;
                      if (_var_121) {
                        continue;
                      }
                    }
                  }
                  break;
                }
                _var_118 = (uint64_t) _var_122;
              }
            }
            generic64_t _var_123;
            _var_27 = _var_118;
            pad(f, (int8_t) 32, (int32_t) (number32_t) _var_50, (int32_t) (number32_t) _var_109, (int32_t) ((number32_t) _var_97 ^ 0x2000));
            _var_123 = (int64_t) ((number64_t) _var_50 << 32) < (int64_t) ((number64_t) _var_109 << 32) ? _var_109 : _var_50;
            _stack.offset_8 = (number32_t) _var_123;
            _var_28 = _var_82;
            _var_29 = _var_85;
          } break;
          case 4206106:
          case 4206193:
          case 4206216:
          case 4206315:
          case 4206395:
          case 4206461:
          case 4206607:
          case 4206659:
          case 4206676:
          case 4206857:
          case 4206893:
          {
            switch ((number64_t) *((generic64_t *) &"\t1@"[8 * ((_var_90 + 4294967231) & 0xFFFFFFFF)])) {
              case 4206106:
              {
                _var_27 = (_var_90 + 4294967231) & 0xFFFFFFFF;
                _var_28 = _var_82;
                _var_29 = _var_85;
                if (_var_80 < 8) {
                  _var_27 = (_var_90 + 4294967231) & 0xFFFFFFFF;
                  _var_28 = _var_82;
                  _var_29 = _var_85;
                  switch ((number64_t) *((generic64_t *) &",.@"[8 * _var_80])) {
                    case 4205394:
                    {
                      break;
                    } break;
                    case 4206124:
                    {
                      _var_27 = _stack.offset_80.offset_0;
                      *((generic32_t *) _var_27) = _stack.offset_4;
                      _var_28 = _var_82;
                      _var_29 = _var_85;
                    } break;
                    case 4206140:
                    {
                      _var_27 = _stack.offset_80.offset_0;
                      *((generic16_t *) _var_27) = (number16_t) _stack.offset_4;
                      _var_28 = _var_82;
                      _var_29 = _var_85;
                    } break;
                    case 4206157:
                    {
                      _var_27 = _stack.offset_80.offset_0;
                      *((generic8_t *) _var_27) = *((generic8_t *) &_stack.offset_4);
                      _var_28 = _var_82;
                      _var_29 = _var_85;
                    } break;
                    case 4206175:
                    {
                      _var_27 = _stack.offset_4;
                      *((generic64_t *) _stack.offset_80.offset_0) = _var_27;
                      _var_28 = _var_82;
                      _var_29 = _var_85;
                    } break;
                    default:
                    {
                    } break;
                  }
                }
              } break;
              case 4206857:
              {
                int32_t _var_124;
                *((generic64_t *) ((pointer_or_number64_t) &_stack - 8)) = _stack.offset_80.offset_8;
                *((generic64_t *) ((pointer_or_number64_t) &_stack - 16)) = _stack.offset_80.offset_0;
                _var_124 = fmt_fp(f, (float128_t) _var_85, (int32_t) (number32_t) _var_50, (int32_t) _var_65, (int32_t) (number32_t) _var_97, (int32_t) (number32_t) _var_90);
                _stack.offset_8 = _var_124;
                _var_27 = *((generic64_t *) ((pointer_or_number64_t) &_stack - 16));
                _var_28 = _var_82;
                _var_29 = _var_85;
              } break;
              case 4206193:
              case 4206216:
              case 4206315:
              case 4206395:
              case 4206461:
              case 4206607:
              case 4206659:
              case 4206676:
              case 4206893:
              {
                switch ((number64_t) *((generic64_t *) &"\t1@"[8 * ((_var_90 + 4294967231) & 0xFFFFFFFF)])) {
                  case 4206607:
                  {
                    *((generic8_t *) ((pointer_or_number64_t) &_stack.offset_80 + 63)) = (number8_t) _stack.offset_80.offset_0;
                    _var_100 = "-+   0X0x";
                    _var_102 = (pointer_or_number64_t) &_stack.offset_80 + 63;
                    _var_98 = 1;
                    _var_99 = _var_49 & 0xFFFEFFFF;
                    _var_101 = (pointer_or_number64_t) &_stack.offset_80 + 64;
                  } break;
                  case 4206659:
                  case 4206676:
                  {
                    generic64_t _var_125;
                    switch ((number64_t) *((generic64_t *) &"\t1@"[8 * ((_var_90 + 4294967231) & 0xFFFFFFFF)])) {
                      case 4206659:
                      {
                        int8_t *_var_126;
                        int32_t *_var_127;
                        _var_127 = unreserved___errno_location();
                        _var_126 = strerror(*_var_127);
                        _var_125 = _var_126;
                      } break;
                      case 4206676:
                      {
                        generic64_t _var_128;
                        _var_128 = !_stack.offset_80.offset_0 ? (generic64_t) "(null)" : _stack.offset_80.offset_0;
                        _var_125 = _var_128;
                      } break;
                    }
                    void *_var_129;
                    *((generic64_t *) &_stack.offset_32) = _var_65;
                    _var_129 = memchr((const void *) _var_125, (int32_t) 0, (size_t) _var_65);
                    if (!_var_129) {
                      _var_100 = "-+   0X0x";
                      _var_101 = &((int8_t *) _var_125)[*((generic64_t *) &_stack.offset_32)];
                      _var_98 = _var_65;
                      _var_99 = _var_49 & 0xFFFEFFFF;
                      _var_102 = _var_125;
                    } else {
                      _var_98 = ((pointer_or_number64_t) _var_129 - _var_125) & 0xFFFFFFFF;
                      _var_100 = "-+   0X0x";
                      _var_99 = _var_49 & 0xFFFEFFFF;
                      _var_101 = _var_129;
                      _var_102 = _var_125;
                    }
                  } break;
                  case 4206193:
                  case 4206216:
                  case 4206315:
                  case 4206395:
                  case 4206461:
                  {
                    generic64_t _var_130;
                    generic64_t _var_131;
                    generic64_t _var_132;
                    generic64_t _var_133;
                    generic64_t _var_134;
                    generic64_t _var_135;
                    generic64_t _var_136;
                    generic64_t _var_137;
                    generic64_t _var_138;
                    switch ((number64_t) *((generic64_t *) &"\t1@"[8 * ((_var_90 + 4294967231) & 0xFFFFFFFF)])) {
                      case 4206315:
                      {
                        generic64_t _var_139;
                        _var_139 = (pointer_or_number64_t) &_stack.offset_80 + 64;
                        if (_stack.offset_80.offset_0) {
                          generic64_t _var_140;
                          generic64_t _var_141;
                          _var_140 = 0;
                          _var_141 = _stack.offset_80.offset_0;
                          generic64_t _var_142;
                          generic64_t _var_143;
                          do {
                            _var_142 = _var_141;
                            _var_143 = (pointer_or_number64_t) &_stack.offset_80 + 63 - _var_140;
                            _var_141 = _var_142 >> 3;
                            *((generic8_t *) _var_143) = ((number8_t) _var_142 & 0x7) | 0x30;
                            _var_140 = _var_140 + 1;
                          } while (!(_var_142 < 8));
                          _var_139 = _var_143;
                        }
                        _var_138 = _var_139;
                        _var_136 = _var_65;
                        _var_137 = _var_97;
                        if ((_var_97 & 0x8)) {
                          _var_136 = _var_65;
                          _var_137 = _var_97;
                          _var_138 = _var_139;
                          if (!((int64_t) ((pointer_or_number64_t) &_stack.offset_80 + 64 - _var_139) < _var_65)) {
                            _var_136 = ((pointer_or_number64_t) &_stack.offset_80 + 64 - _var_139 + 1) & 0xFFFFFFFF;
                            _var_137 = _var_97;
                            _var_138 = _var_139;
                          }
                        }
                        _var_132 = _var_136;
                        _var_133 = _var_137;
                        _var_135 = _var_138;
                        _var_131 = _var_83 & 0xFFFFFFFF;
                        _var_134 = "-+   0X0x";
                        _var_130 = _var_83;
                      } break;
                      case 4206395:
                      case 4206461:
                      {
                        generic64_t _var_144;
                        uint8_t *_var_145;
                        switch ((number64_t) *((generic64_t *) &"\t1@"[8 * ((_var_90 + 4294967231) & 0xFFFFFFFF)])) {
                          case 4206395:
                          {
                            if ((int64_t) _stack.offset_80.offset_0 > -1) {
                              generic64_t _var_146;
                              _var_146 = _lshift(_stack.offset_80.offset_0, 4294967240);
                              _var_145 = "+   0X0x";
                              _var_144 = 1;
                              if (!(_var_97 & 0x800)) {
                                uint8_t *_var_147;
                                generic64_t _var_148;
                                _var_148 = !(_var_97 & 0x1) ? _var_83 & 0xFFFFFFFF : 1;
                                _var_144 = _var_148;
                                _var_147 = !(_var_97 & 0x1) ? (generic64_t) "-+   0X0x" : (generic64_t) "   0X0x";
                                _var_145 = _var_147;
                              }
                            } else {
                              _stack.offset_80.offset_0 = 0 - _stack.offset_80.offset_0;
                              _var_145 = "-+   0X0x";
                              _var_144 = 1;
                            }
                          } break;
                          case 4206461:
                          {
                            _var_144 = _var_83 & 0xFFFFFFFF;
                            _var_145 = "-+   0X0x";
                          } break;
                        }
                        int8_t *_var_149;
                        _var_131 = _var_144;
                        *((generic64_t *) &_stack.offset_32) = _var_83;
                        *((uint8_t **) ((pointer_or_number64_t) &_stack.offset_4 + 4)) = _var_145;
                        _var_149 = fmt_u(_stack.offset_80.offset_0, (int8_t *) ((pointer_or_number64_t) &_stack.offset_80 + 64));
                        _var_135 = _var_149;
                        _var_134 = *((generic64_t *) ((pointer_or_number64_t) &_stack.offset_4 + 4));
                        _var_130 = *((generic64_t *) &_stack.offset_32);
                        _var_132 = _var_65;
                        _var_133 = _var_97;
                      } break;
                      case 4206193:
                      case 4206216:
                      {
                        if ((number64_t) *((generic64_t *) &"\t1@"[8 * ((_var_90 + 4294967231) & 0xFFFFFFFF)]) == 4206193) {
                          _var_103 = _llvm_umax_i32(_var_65, 16);
                          _var_104 = (_var_97 & 0xFFFFFFF7) | 0x8;
                          _var_105 = 120;
                        }
                        generic64_t _var_150;
                        _var_150 = (pointer_or_number64_t) &_stack.offset_80 + 64;
                        if (_stack.offset_80.offset_0) {
                          generic64_t _var_151;
                          generic64_t _var_152;
                          generic64_t _var_153;
                          _var_151 = 0;
                          _var_152 = _var_84;
                          _var_153 = _stack.offset_80.offset_0;
                          generic64_t _var_154;
                          generic64_t _var_155;
                          do {
                            _var_154 = _var_153;
                            _var_155 = (pointer_or_number64_t) &_stack.offset_80 + 63 - _var_151;
                            _var_153 = _var_154 >> 4;
                            _var_152 = (_var_152 & 0xFFFFFF00) | *((generic8_t *) ((_var_154 & 0xF) | (number64_t) "0123456789ABCDEF")) | (_var_105 & 0x20);
                            *((generic8_t *) _var_155) = (number8_t) _var_152;
                            _var_151 = _var_151 + 1;
                          } while (!(_var_154 < 16));
                          _var_150 = _var_155;
                        }
                        _var_138 = _var_150;
                        _var_136 = _var_103;
                        _var_137 = _var_104;
                        if (!_stack.offset_80.offset_0 || !(_var_104 & 0x8)) {
                          _var_132 = _var_136;
                          _var_133 = _var_137;
                          _var_135 = _var_138;
                          _var_131 = _var_83 & 0xFFFFFFFF;
                          _var_134 = "-+   0X0x";
                          _var_130 = _var_83;
                        } else {
                          _var_134 = ((int32_t) (number32_t) _var_105 >> 4) + 4215957;
                          _var_130 = _var_83;
                          _var_131 = 2;
                          _var_132 = _var_103;
                          _var_133 = _var_104;
                          _var_135 = _var_150;
                        }
                      } break;
                    }
                    generic64_t _var_156;
                    _var_156 = (int32_t) (number32_t) _var_132 < (int32_t) 0 ? _var_133 : _var_133 & 0xFFFEFFFF;
                    if (!(number32_t) _var_132 && !_stack.offset_80.offset_0) {
                      _stack.offset_8 = (number32_t) _var_131;
                      _var_98 = _var_130 & 0xFFFFFFFF;
                      _var_102 = (pointer_or_number64_t) &_stack.offset_80 + 64;
                      _var_99 = _var_156;
                      _var_100 = _var_134;
                      _var_101 = (pointer_or_number64_t) &_stack.offset_80 + 64;
                    } else {
                      generic64_t _var_157;
                      _var_157 = !_stack.offset_80.offset_0;
                      _stack.offset_8 = (number32_t) _var_131;
                      _var_98 = _llvm_smax_i64((pointer_or_number64_t) &_stack.offset_80 + 64 - _var_135 + _var_157, (int64_t) ((number64_t) _var_132 << 32) >> 32);
                      _var_99 = _var_156;
                      _var_100 = _var_134;
                      _var_101 = (pointer_or_number64_t) &_stack.offset_80 + 64;
                      _var_102 = _var_135;
                    }
                  } break;
                }
                *((generic64_t *) &_stack.offset_32) = _var_101 - _var_102;
                _var_27 = _stack.offset_8;
                _var_96 = (int64_t) ((int64_t) ((number64_t) _var_98 << 32) >> 32) < (int64_t) (_var_101 - _var_102) ? _var_101 - _var_102 : _var_98;
                _stack.offset_56 = _var_100;
                _stack.offset_52 = (number32_t) ((_var_96 & 0xFFFFFFFF) + _var_27);
                _var_95 = (int64_t) ((number64_t) ((_var_96 & 0xFFFFFFFF) + _var_27) << 32) < (int64_t) ((number64_t) _var_50 << 32) ? _var_50 : (_var_96 & 0xFFFFFFFF) + _var_27;
                pad(f, (int8_t) 32, (int32_t) (number32_t) _var_95, (int32_t) (number32_t) ((_var_96 & 0xFFFFFFFF) + _var_27), (int32_t) (number32_t) _var_99);
                out(f, (const int8_t *) &_stack.offset_56->member_0.member_1, (size_t) _stack.offset_8);
                _stack.offset_8 = _stack.offset_52;
                pad(f, (int8_t) 48, (int32_t) (number32_t) _var_95, (int32_t) _stack.offset_52, (int32_t) ((number32_t) _var_99 ^ 0x10000));
                pad(f, (int8_t) 48, (int32_t) (number32_t) _var_96, (int32_t) _stack.offset_32.member_1, (int32_t) 0);
                out(f, (const int8_t *) _var_102, *((generic64_t *) &_stack.offset_32));
                pad(f, (int8_t) 32, (int32_t) (number32_t) _var_95, (int32_t) _stack.offset_8, (int32_t) ((number32_t) _var_99 ^ 0x2000));
                _stack.offset_8 = (number32_t) _var_95;
                _var_28 = _var_82;
                _var_29 = _var_85;
              } break;
            }
          } break;
          default:
          {
          } break;
        }
        if (_break_from_loop_108)
          break;
        continue;
      }
      _var_1 = _var_48;
    }
    break;
  }
  switch (_loop_state_var) {
    case 0:
    case 1:
    case 3:
    {
      generic8_t _var_158;
      generic64_t _var_159;
      generic32_t _var_160;
      generic64_t _var_161;
      generic64_t _var_162;
      if (!(_loop_state_var)) {
        while (true) {
          generic64_t _var_163;
          _var_163 = _var_4;
          pop_arg((arg *) ((_var_6 << 4) + *((generic64_t *) &_stack.offset_40)), (int32_t) _var_5, (va_list *) *((generic64_t *) &_stack.offset_16));
          if (_var_6 + 1 == 10) {
            _var_1 = 1;
            _stack.offset_4 = _var_1;
            return (int32_t) _stack.offset_4;
          }
          _var_5 = *((generic32_t *) (((_var_6 + 1) << 2) + (pointer_or_number64_t) _stack.offset_24));
          _var_4 = _var_163 + 1;
          _var_6 = _var_6 + 1;
          if (_var_5) {
            continue;
          }
          _var_2 = (int64_t) (((number64_t) _var_163 << 32) + 8589934592) >> 32;
          _var_3 = _stack.offset_24;
          break;
        }
        _var_162 = _var_2;
        _var_159 = (pointer_or_number64_t) _var_3 + (_var_162 << 2);
        _var_161 = 0;
        while (true) {
          _var_160 = 1;
          if (!(_var_162 > 9)) {
            _var_162 = _var_162 + 1;
            _var_158 = !*((generic32_t *) (_var_159 + (_var_161 << 2)));
            _var_161 = _var_161 + 1;
            _var_160 = 4294967295;
            if (_var_158) {
              continue;
            }
          }
          break;
        }
        _var_1 = _var_160;
      } else {
        _var_162 = _var_2;
        _var_159 = (pointer_or_number64_t) _var_3 + (_var_162 << 2);
        _var_161 = 0;
        while (true) {
          _var_160 = 1;
          if (!(_var_162 > 9)) {
            _var_162 = _var_162 + 1;
            _var_158 = !*((generic32_t *) (_var_159 + (_var_161 << 2)));
            _var_161 = _var_161 + 1;
            _var_160 = 4294967295;
            if (_var_158) {
              continue;
            }
          }
          break;
        }
        _var_1 = _var_160;
      }
      _stack.offset_4 = _var_1;
    } break;
  }
  return (int32_t) _stack.offset_4;
}

_ABI(SystemV_x86_64)
int32_t vfprintf(typedef_88 f, typedef_104 fmt, unreserved___va_list_tag *ap) {
  struct _PACKED struct_566 {
    uint8_t padding_at_0[12];
    generic32_t offset_12;
    union_640 offset_16;
    uint8_t padding_at_40[336];
  } _stack;
  generic64_t _var_0;
  generic64_t _var_1;
  _var_1 = (pointer_or_number64_t) &_stack.offset_16.member_2.offset_16 + 8;
  _var_0 = 0;
  do {
    *((generic32_t *) _var_1) = 0;
    _var_1 = _var_1 + 4;
    _var_0 = _var_0 + 1;
  } while (_var_0 != 10);
  int32_t _var_2;
  generic64_t _var_3;
  _stack.offset_16.member_3 = *((generic64_t *) ap);
  _stack.offset_16.member_1.offset_8 = ap->overflow_arg_area;
  _stack.offset_16.member_2.offset_16 = ap->reg_save_area;
  _var_2 = printf_core((FILE_ *) NULL, fmt, (va_list *) &_stack.offset_16, (arg *) ((pointer_or_number64_t) &_stack.offset_16 + 144), (int32_t *) ((pointer_or_number64_t) &_stack.offset_16 + 24));
  _var_3 = 4294967295;
  if (_var_2 > -1) {
    generic64_t _var_4;
    _var_4 = 0;
    if (f->lock > -1) {
      int32_t _var_5;
      _var_5 = unreserved___lockfile(f);
      _var_4 = (uint64_t) _var_5;
    }
    _stack.offset_12 = f->flags & 0x20;
    if (!(f->mode > (int8_t) 0)) {
      f->flags = f->flags & 0xFFFFFFDF;
    }
    generic64_t _var_6;
    _var_6 = 0;
    if (!f->buf_size) {
      _var_6 = f->buf;
      f->buf_size = 80;
      f->buf = (pointer_or_number64_t) &_stack.offset_16 + 64;
      f->wbase = (pointer_or_number64_t) &_stack.offset_16 + 64;
      f->wpos = (pointer_or_number64_t) &_stack.offset_16 + 64;
      f->wend = (pointer_or_number64_t) &_stack.offset_16 + 144;
    }
    int32_t _var_7;
    generic64_t _var_8;
    _var_7 = printf_core(f, fmt, (va_list *) &_stack.offset_16, (arg *) ((pointer_or_number64_t) &_stack.offset_16 + 144), (int32_t *) ((pointer_or_number64_t) &_stack.offset_16 + 24));
    _var_8 = (uint64_t) _var_7;
    if (_var_6) {
      generic64_t _var_9;
      pointer_or_number64_t _var_10;
      pointer_or_number64_t _var_11;
      artificial_struct_returned_by_rawfunction_25 _var_12;
      _var_12 = ((rawfunction_25 *) f->write)((pointer_or_number64_t) &_stack.offset_16 + 144, 0, 0, (pointer_or_number64_t) f, (pointer_or_number64_t) &_stack.offset_16.member_2.offset_16 + 8, _undef_generic64_t());
      _var_11 = _var_12.register_rax;
      _var_10 = _var_12.register_rdx;
      f->buf = _var_6;
      f->buf_size = 0;
      _var_9 = !f->wpos ? 4294967295 : (uint64_t) _var_7;
      _var_8 = _var_9;
      f->wend = 0;
      f->wbase = 0;
      f->wpos = 0;
    }
    generic64_t _var_13;
    _var_13 = !(f->flags & 0x20) ? _var_8 : 4294967295;
    _var_3 = _var_13;
    f->flags = _stack.offset_12 | f->flags;
    if (_var_4) {
      unreserved___unlockfile(f);
      _var_3 = _var_13;
    }
  }
  return (int32_t) (number32_t) _var_3;
}

_ABI(SystemV_x86_64)
void *memchr(const void *src, int32_t c, size_t n) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  _var_1 = n;
  _var_2 = src;
  if (((number64_t) src & 0x7)) {
    _var_0 = 0;
    if (!n) {
      return (void *) _var_0;
    }
    generic64_t _var_3;
    generic64_t _var_4;
    _var_3 = 0;
    _var_4 = src;
    while (true) {
      generic64_t _var_5;
      _var_5 = _var_4;
      if (*((generic8_t *) _var_5) != (number8_t) c) {
        if (!(((pointer_or_number64_t) src + 1 + _var_3) & 0x7)) {
          _var_1 = n + (pointer_or_number64_t) src - ((pointer_or_number64_t) src + 1 + _var_3);
          _var_2 = (pointer_or_number64_t) src + 1 + _var_3;
          break;
        }
        generic8_t _var_6;
        _var_4 = _var_4 + 1;
        _var_6 = n + (pointer_or_number64_t) src == (pointer_or_number64_t) src + 1 + _var_3;
        _var_3 = _var_3 + 1;
        _var_5 = 0;
        if (!(_var_6)) {
          continue;
        }
      }
      _var_0 = _var_5;
      return (void *) _var_0;
    }
  }
  _var_0 = 0;
  if (_var_1) {
    _var_0 = _var_2;
    if (*((generic8_t *) _var_2) != (number8_t) c) {
      generic64_t _var_7;
      generic64_t _var_8;
      _var_7 = _var_1;
      _var_8 = _var_2;
      if (_var_1 > 7) {
        generic64_t _var_9;
        generic64_t _var_10;
        generic64_t _var_11;
        _var_9 = 0;
        _var_10 = _var_1;
        _var_11 = _var_2;
        while (true) {
          if (!(((*((generic64_t *) _var_11) ^ (((number32_t) c & 0xFF) * 72340172838076673)) - 72340172838076673) & ~(*((generic64_t *) _var_11) ^ (((number32_t) c & 0xFF) * 72340172838076673)) & 0x8080808080808080)) {
            generic64_t _var_12;
            _var_12 = _var_9 << 3;
            _var_10 = _var_10 - 8;
            _var_11 = _var_11 + 8;
            _var_9 = _var_9 + 1;
            if (_var_1 - 8 - _var_12 > 7) {
              continue;
            }
            _var_8 = _var_2 + 8 + _var_12;
            _var_0 = 0;
            _var_7 = _var_1 - 8 - _var_12;
            if (!(_var_1 - 8 - _var_12)) {
              return (void *) _var_0;
            }
          } else {
            _var_7 = _var_10;
            _var_8 = _var_11;
          }
          break;
        }
      }
      generic64_t _var_13;
      generic64_t _var_14;
      generic64_t _var_15;
      generic64_t _var_16;
      _var_16 = _var_8;
      _var_14 = _var_16 + _var_7;
      _var_13 = _var_16 + 1;
      _var_15 = 0;
      generic64_t _var_17;
      while (true) {
        _var_17 = _var_16;
        if (*((generic8_t *) _var_17) != (number8_t) c) {
          generic8_t _var_18;
          _var_16 = _var_16 + 1;
          _var_18 = _var_13 + _var_15 == _var_14;
          _var_15 = _var_15 + 1;
          _var_17 = 0;
          if (!(_var_18)) {
            continue;
          }
        }
        break;
      }
      _var_0 = _var_17;
    }
  }
  return (void *) _var_0;
}

_ABI(SystemV_x86_64)
struct_661 *memset(struct_661 *argument_0, generic64_t argument_1, generic64_t argument_2, generic64_t argument_3, generic64_t argument_4) {
  if (argument_2 > 126) {
    generic64_t _var_0;
    generic64_t _var_1;
    *((generic64_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 8)) = (argument_1 & 0xFF) * 72340172838076673;
    _var_0 = argument_2;
    _var_1 = argument_0;
    if (((number64_t) argument_0 & 0xF)) {
      argument_0->offset_0.member_0.member_0.member_3 = (argument_1 & 0xFF) * 72340172838076673;
      argument_0->offset_0.member_0.member_0.member_5.offset_8 = (argument_1 & 0xFF) * 72340172838076673;
      _var_0 = argument_2 - ((0 - (number64_t) argument_0) & 0xF);
      _var_1 = ((0 - (number64_t) argument_0) & 0xF) + (pointer_or_number64_t) argument_0;
    }
    if (!(_var_0 < 8)) {
      generic64_t _var_2;
      generic64_t _var_3;
      _var_2 = 0;
      _var_3 = _var_1;
      do {
        _var_2 = _var_2 + 1;
        ((struct_661 *) _var_3)->offset_0.member_0.member_0.member_3 = (argument_1 & 0xFF) * 72340172838076673;
        _var_3 = &((struct_661 *) _var_3)->offset_0.member_0.member_0.member_5.offset_8;
      } while (_var_0 >> 3 != _var_2);
    }
  } else {
    if ((argument_2 & 0xFFFFFFFF)) {
      argument_0->offset_0.member_0.member_0.member_1 = (number8_t) argument_1;
      *((generic8_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 1)) = (number8_t) argument_1;
      if ((argument_2 & 0xFFFFFFFF) > 2) {
        argument_0->offset_0.member_0.member_0.member_0.offset_1.member_0.member_3 = (number16_t) ((argument_1 & 0xFF) * 72340172838076673);
        *((generic16_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 3)) = (number16_t) ((argument_1 & 0xFF) * 72340172838076673);
        if ((argument_2 & 0xFFFFFFFF) > 6) {
          argument_0->offset_0.member_0.member_0.member_0.offset_1.member_0.member_5.offset_2 = (number32_t) ((argument_1 & 0xFF) * 72340172838076673);
          *((generic32_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 7)) = (number32_t) ((argument_1 & 0xFF) * 72340172838076673);
          if ((argument_2 & 0xFFFFFFFF) > 14) {
            argument_0->offset_0.member_0.member_0.member_4.offset_7 = (argument_1 & 0xFF) * 72340172838076673;
            *((generic64_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 15)) = (argument_1 & 0xFF) * 72340172838076673;
            if ((argument_2 & 0xFFFFFFFF) > 30) {
              argument_0->offset_0.member_0.member_2.offset_15.member_1 = (argument_1 & 0xFF) * 72340172838076673;
              argument_0->offset_0.member_0.member_2.offset_15.member_2.offset_8 = (argument_1 & 0xFF) * 72340172838076673;
              *((generic64_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 31)) = (argument_1 & 0xFF) * 72340172838076673;
              *((generic64_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 23)) = (argument_1 & 0xFF) * 72340172838076673;
              if ((argument_2 & 0xFFFFFFFF) > 62) {
                argument_0->offset_0.member_1.offset_31 = (argument_1 & 0xFF) * 72340172838076673;
                argument_0->offset_39 = (argument_1 & 0xFF) * 72340172838076673;
                argument_0->offset_47 = (argument_1 & 0xFF) * 72340172838076673;
                argument_0->offset_55 = (argument_1 & 0xFF) * 72340172838076673;
                *((generic64_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 63)) = (argument_1 & 0xFF) * 72340172838076673;
                *((generic64_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 55)) = (argument_1 & 0xFF) * 72340172838076673;
                *((generic64_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 47)) = (argument_1 & 0xFF) * 72340172838076673;
                *((generic64_t *) (argument_2 + (pointer_or_number64_t) argument_0 - 39)) = (argument_1 & 0xFF) * 72340172838076673;
              }
            }
          }
        }
      }
    }
  }
  return argument_0;
}

_ABI(SystemV_x86_64)
size_t strlen(const int8_t *s) {
  generic64_t _var_0;
  const int8_t *_var_1;
  _var_1 = s;
  if (((number64_t) s & 0x7)) {
    generic64_t _var_2;
    const int8_t *_var_3;
    _var_2 = 0;
    _var_3 = s;
    while (true) {
      generic64_t _var_4;
      _var_4 = _var_2;
      if (!*_var_3) {
        _var_0 = _var_3;
        return _var_0 - (number64_t) s;
      }
      _var_2 = _var_4 + 1;
      _var_3 = &_var_3[1];
      if (((number64_t) &s[_var_4 + 1] & 0x7)) {
        continue;
      }
      _var_1 = &s[_var_4 + 1];
      break;
    }
  }
  generic64_t _var_5;
  _var_5 = _var_1;
  if (!((*((generic64_t *) _var_5) - 72340172838076673) & ~*((generic64_t *) _var_5) & 0x8080808080808080)) {
    generic64_t _var_6;
    _var_6 = 0;
    generic64_t _var_7;
    do {
      _var_7 = _var_6;
      _var_6 = _var_7 + 1;
    } while (!((*((generic64_t *) &_var_1[8 * _var_7 + 8]) - 72340172838076673) & ~*((generic64_t *) &_var_1[8 * _var_7 + 8]) & 0x8080808080808080));
    _var_5 = &_var_1[8 * _var_7 + 8];
  }
  generic64_t _var_8;
  _var_8 = _var_5;
  generic64_t _var_9;
  do {
    _var_9 = _var_8;
    _var_8 = &((const int8_t *) _var_9)[1];
  } while (*((generic8_t *) _var_9));
  _var_0 = _var_9;
  return _var_0 - (number64_t) s;
}

_ABI(SystemV_x86_64)
void *unreserved___copy_tls(uint8_t *mem) {
  generic64_t _var_0;
  _var_0 = mem;
  if (segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_0) {
    struct_718 *_var_1;
    *((generic64_t *) mem) = 1;
    _var_0 = ((pointer_or_number64_t) &mem[segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_24] - 336) & (0 - segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_24);
    *((uint8_t **) (_var_0 + 8)) = mem;
    *((uint8_t **) (_var_0 + 328)) = mem;
    *((generic64_t *) &mem[8]) = _var_0 - segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16;
    _var_1 = memcpy((struct_718 *) (_var_0 - segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16), segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_0, segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_8);
  }
  return (void *) _var_0;
}

_ABI(SystemV_x86_64)
void unreserved___init_tls(size_t *aux) {
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  generic32_t _var_24;
  generic64_t _var_25;
  generic64_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic32_t _var_29;
  generic64_t _var_30;
  generic32_t _var_31;
  generic32_t _var_32;
  generic32_t _var_33;
  generic64_t _var_34;
  generic32_t _var_35;
  generic32_t _var_36;
  generic32_t _var_37;
  generic64_t _var_38;
  generic32_t _var_39;
  generic32_t _var_40;
  generic64_t _var_41;
  generic32_t _var_42;
  generic32_t _var_43;
  generic32_t _var_44;
  generic64_t _var_45;
  generic32_t _var_46;
  generic8_t _var_47;
  generic64_t _var_48;
  generic64_t _var_49;
  generic64_t _var_50;
  _var_48 = 0;
  _var_49 = 0;
  if (aux[5]) {
    generic64_t _var_51;
    generic64_t _var_52;
    generic64_t _var_53;
    generic64_t _var_54;
    _var_51 = 0;
    _var_52 = 0;
    _var_53 = 0;
    _var_54 = aux[3];
    generic64_t _var_55;
    generic8_t _var_56;
    generic64_t _var_57;
    generic64_t _var_58;
    do {
      _var_55 = *((generic32_t *) _var_54);
      if (*((generic32_t *) _var_54) == 6) {
        _var_58 = aux[3] - *((generic64_t *) (_var_54 + 16));
        _var_57 = _var_52;
      } else {
        generic64_t _var_59;
        _var_59 = *((generic32_t *) _var_54) == 7 ? _var_54 : _var_52;
        _var_57 = _var_59;
        _var_58 = _var_53;
      }
      _var_54 = _var_54 + aux[4];
      _var_56 = aux[5] == _var_51 + 1;
      _var_51 = _var_51 + 1;
    } while (!(_var_56));
    _var_48 = _var_58;
    _var_49 = _var_57;
    _var_50 = _var_55;
  }
  generic64_t _var_60;
  if (!_var_49) {
    _var_60 = segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_24;
  } else {
    segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_0 = _var_48 + *((generic64_t *) (_var_49 + 16));
    segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_8 = *((generic64_t *) (_var_49 + 32));
    _var_60 = *((generic64_t *) (_var_49 + 48));
    segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16 = *((generic64_t *) (_var_49 + 40));
    segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_24 = _var_60;
  }
  segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16 = ((_var_60 - 1) & (0 - ((pointer_or_number64_t) segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_0 + segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16))) + segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16;
  if (!(_var_60 > 7)) {
    segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_24 = 8;
  }
  generic32_t _var_61;
  generic64_t _var_62;
  generic32_t _var_63;
  generic64_t _var_64;
  generic64_t _var_65;
  generic64_t _var_66;
  generic64_t _var_67;
  generic64_t _var_68;
  generic32_t _var_69;
  generic64_t _var_70;
  generic64_t _var_71;
  generic32_t _var_72;
  generic64_t _var_73;
  generic32_t _var_74;
  _var_64 = segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_24;
  segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_24 = (_var_64 + (((_var_60 - 1) & (0 - ((pointer_or_number64_t) segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_0 + segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16))) + segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16) + 359) & 0xFFFFFFFFFFFFFFF8;
  _var_65 = &segment_0x406fd0_Generic64_2224.unreserved__bss.builtin_tls_;
  _var_61 = 4294967295;
  _var_62 = 514;
  _var_63 = 4243635;
  _var_66 = 0;
  _var_67 = 0;
  _var_68 = 0;
  _var_69 = 65535;
  _var_70 = _var_50;
  _var_71 = aux[3];
  _var_72 = 0;
  _var_73 = 0;
  _var_74 = 4294967295;
  if (((_var_64 + (((_var_60 - 1) & (0 - ((pointer_or_number64_t) segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_0 + segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16))) + segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16) + 359) & 0xFFFFFFFFFFFFFFF8) > 472) {
    _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10233, 34, 18446744073709551615U, 0, 9, _undef_generic64_t(), _undef_generic64_t(), segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16, 0, 3, (_var_64 + (((_var_60 - 1) & (0 - ((pointer_or_number64_t) segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_0 + segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16))) + segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16) + 359) & 0xFFFFFFFFFFFFFFF8, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32, &_var_33, &_var_34, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42, &_var_43, &_var_44, &_var_45, &_var_46, &_var_47);
    _var_65 = _var_27;
    _var_61 = _var_29;
    _var_62 = _var_30;
    _var_63 = _var_32;
    _var_66 = _var_34;
    _var_67 = _var_38;
    _var_68 = _var_41;
    _var_69 = _var_42;
    _var_72 = _var_43;
    _var_73 = _var_45;
    _var_74 = _var_46;
    _var_64 = 3;
    _var_70 = 0;
    _var_71 = 18446744073709551615U;
  }
  generic64_t _var_75;
  generic64_t _var_76;
  void *_var_77;
  _var_77 = unreserved___copy_tls((uint8_t *) _var_65);
  *((void **) _var_77) = _var_77;
  _var_76 = unreserved___set_thread_area((struct_703 *) _var_77);
  _var_75 = _lshift(_var_76 & 0xFFFFFFFF, 4294967272);
  if (!(number32_t) _var_76) {
    segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_0 = 1;
  }
  _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10285, 34, _var_71, _var_70, 218, _undef_generic64_t(), _var_77, segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___static_tls.offset_16, (pointer_or_number64_t) _var_77 + 56, _var_64, segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_24, _var_61, _var_62, _var_63, 0, 0, 15727360, 0, 13628160, 0, _var_66, _var_67, _var_68, _var_69, 274877906944, 127, 2147549185, 0, _var_72, _var_73, _var_74, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
  *((generic32_t *) ((pointer_or_number64_t) _var_77 + 56)) = (number32_t) _var_3;
  *((generic64_t *) ((pointer_or_number64_t) _var_77 + 256)) = (pointer_or_number64_t) &segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_32 + 8;
  *((generic64_t *) ((pointer_or_number64_t) _var_77 + 224)) = (pointer_or_number64_t) _var_77 + 224;
}

_ABI(SystemV_x86_64)
int32_t *unreserved___errno_location(void) {
  return (int32_t *) (*((generic64_t *) NULL) + 68);
}

_ABI(SystemV_x86_64)
int8_t *strerror_l(int32_t e, locale_t_ loc) {
  generic64_t _var_0;
  generic8_t _var_1;
  generic32_t _var_2;
  _var_1 = segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_0;
  _var_0 = 0;
  _var_2 = 0;
  if (_var_1) {
    _var_0 = 0;
    _var_1 = segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_0;
    _var_2 = e;
    if (segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_0 != (pointer_or_number32_t) e) {
      generic64_t _var_3;
      _var_3 = 0;
      generic64_t _var_4;
      while (true) {
        _var_4 = _var_3;
        if (*((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_1 + _var_4 * 1))) {
          _var_3 = _var_4 + 1;
          if (*((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_1 + _var_4 * 1)) != (pointer_or_number32_t) e) {
            continue;
          }
        }
        break;
      }
      _var_0 = _var_4 + 1;
      _var_1 = *((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_1 + _var_4 * 1));
      _var_2 = e;
    }
  }
  uint8_t *_var_5;
  _var_5 = "Illegal byte sequence";
  if ((_var_0 & 0xFFFFFFFF)) {
    generic64_t _var_6;
    uint8_t *_var_7;
    generic64_t _var_8;
    generic64_t _var_9;
    _var_8 = _var_1;
    _var_9 = _var_2;
    _var_7 = "Illegal byte sequence";
    _var_6 = _var_0;
    generic64_t _var_10;
    uint8_t *_var_11;
    generic64_t _var_12;
    do {
      generic64_t _var_13;
      generic32_t _var_14;
      generic64_t _var_15;
      generic64_t _var_16;
      generic64_t _var_17;
      _var_11 = _var_7;
      _var_17 = _var_11;
      _var_16 = _var_8;
      _var_15 = _var_9;
      _var_12 = _var_6 & 0xFFFFFFFF;
      _var_13 = 0;
      _var_14 = 24;
      generic8_t _var_18;
      generic64_t _var_19;
      do {
        generic64_t _var_20;
        _var_10 = _var_13;
        _var_19 = (_var_16 & 0xFFFFFFFFFFFFFF00) | *((generic8_t *) _var_17);
        _var_20 = 0;
        switch ((number32_t) _var_14) {
          case 26:
          case 28:
          case 30:
          {
            _var_20 = _var_15;
          } break;
          case 20:
          {
            _var_20 = ~(number32_t) _var_15 < (number32_t) _var_12;
          } break;
          case 18:
          {
            _var_20 = ((number32_t) _var_15 & 0xFF) > (((number32_t) _var_12 + (number32_t) _var_15) & 0xFF);
          } break;
          case 16:
          {
            _var_20 = ~(number32_t) _var_15 < (number32_t) _var_12;
          } break;
        }
        _var_18 = !*((generic8_t *) _var_17);
        _var_13 = _var_10 + 1;
        _var_17 = &((uint8_t *) _var_17)[1];
        _var_14 = 22;
        _var_12 = _var_19;
        _var_16 = _var_19;
      } while (!(_var_18));
      _var_6 = (_var_6 & 0xFFFFFFFF) - 1;
      _var_7 = &_var_11[_var_10 + 1];
      _var_8 = _var_12;
      _var_9 = 0;
    } while ((_var_6 & 0xFFFFFFFF));
    _var_5 = &_var_11[_var_10 + 1];
  }
  return (int8_t *) _var_5;
}

_ABI(SystemV_x86_64)
int8_t *strerror(int32_t e) {
  generic64_t _var_0;
  generic8_t _var_1;
  generic32_t _var_2;
  _var_1 = segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_0;
  _var_0 = 0;
  _var_2 = 0;
  if (_var_1) {
    _var_0 = 0;
    _var_1 = segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_0;
    _var_2 = e;
    if (segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_0 != (pointer_or_number32_t) e) {
      generic64_t _var_3;
      _var_3 = 0;
      generic64_t _var_4;
      while (true) {
        _var_4 = _var_3;
        if (*((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_1 + _var_4 * 1))) {
          _var_3 = _var_4 + 1;
          if (*((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_1 + _var_4 * 1)) != (pointer_or_number32_t) e) {
            continue;
          }
        }
        break;
      }
      _var_0 = _var_4 + 1;
      _var_1 = *((generic8_t *) ((pointer_or_number64_t) &segment_0x405000_Generic64_3292.unreserved__rodata.errid.offset_1 + _var_4 * 1));
      _var_2 = e;
    }
  }
  uint8_t *_var_5;
  _var_5 = "Illegal byte sequence";
  if ((_var_0 & 0xFFFFFFFF)) {
    generic64_t _var_6;
    uint8_t *_var_7;
    generic64_t _var_8;
    generic64_t _var_9;
    _var_8 = _var_1;
    _var_9 = _var_2;
    _var_7 = "Illegal byte sequence";
    _var_6 = _var_0;
    generic64_t _var_10;
    uint8_t *_var_11;
    generic64_t _var_12;
    do {
      generic64_t _var_13;
      generic32_t _var_14;
      generic64_t _var_15;
      generic64_t _var_16;
      generic64_t _var_17;
      _var_11 = _var_7;
      _var_17 = _var_11;
      _var_16 = _var_8;
      _var_15 = _var_9;
      _var_12 = _var_6 & 0xFFFFFFFF;
      _var_13 = 0;
      _var_14 = 24;
      generic8_t _var_18;
      generic64_t _var_19;
      do {
        generic64_t _var_20;
        _var_10 = _var_13;
        _var_19 = (_var_16 & 0xFFFFFFFFFFFFFF00) | *((generic8_t *) _var_17);
        _var_20 = 0;
        switch ((number32_t) _var_14) {
          case 26:
          case 28:
          case 30:
          {
            _var_20 = _var_15;
          } break;
          case 20:
          {
            _var_20 = ~(number32_t) _var_15 < (number32_t) _var_12;
          } break;
          case 18:
          {
            _var_20 = ((number32_t) _var_15 & 0xFF) > (((number32_t) _var_12 + (number32_t) _var_15) & 0xFF);
          } break;
          case 16:
          {
            _var_20 = ~(number32_t) _var_15 < (number32_t) _var_12;
          } break;
        }
        _var_18 = !*((generic8_t *) _var_17);
        _var_13 = _var_10 + 1;
        _var_17 = &((uint8_t *) _var_17)[1];
        _var_14 = 22;
        _var_12 = _var_19;
        _var_16 = _var_19;
      } while (!(_var_18));
      _var_6 = (_var_6 & 0xFFFFFFFF) - 1;
      _var_7 = &_var_11[_var_10 + 1];
      _var_8 = _var_12;
      _var_9 = 0;
    } while ((_var_6 & 0xFFFFFFFF));
    _var_5 = &_var_11[_var_10 + 1];
  }
  return (int8_t *) _var_5;
}

_ABI(SystemV_x86_64) _Noreturn
void unreserved__Exit(int32_t ec) {
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  generic32_t _var_24;
  generic64_t _var_25;
  generic64_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic32_t _var_29;
  generic64_t _var_30;
  generic32_t _var_31;
  generic32_t _var_32;
  generic32_t _var_33;
  generic64_t _var_34;
  generic32_t _var_35;
  generic32_t _var_36;
  generic32_t _var_37;
  generic64_t _var_38;
  generic32_t _var_39;
  generic32_t _var_40;
  generic64_t _var_41;
  generic32_t _var_42;
  generic32_t _var_43;
  generic32_t _var_44;
  generic64_t _var_45;
  generic32_t _var_46;
  generic8_t _var_47;
  generic64_t _var_48;
  generic64_t _var_49;
  generic64_t _var_50;
  generic64_t _var_51;
  generic64_t _var_52;
  generic64_t _var_53;
  generic64_t _var_54;
  generic64_t _var_55;
  generic64_t _var_56;
  generic64_t _var_57;
  _var_57 = &_var_46;
  _var_56 = &_var_45;
  _var_55 = &_var_43;
  _var_54 = &_var_42;
  _var_53 = &_var_41;
  _var_52 = &_var_38;
  _var_51 = &_var_34;
  _var_50 = &_var_29;
  _var_49 = &_var_30;
  _var_48 = &_var_32;
  _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10410, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 231, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), (int64_t) ec, 60, _undef_generic64_t(), 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, _var_50, _var_49, &_var_31, _var_48, &_var_33, _var_51, &_var_35, &_var_36, &_var_37, _var_52, &_var_39, &_var_40, _var_53, _var_54, _var_55, &_var_44, _var_56, _var_57, &_var_47);
  while (true) {
    _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10420, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 60, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), (int64_t) ec, 60, _undef_generic64_t(), *((generic32_t *) _var_50), *((generic64_t *) _var_49), *((generic32_t *) _var_48), 0, 0, 15727360, 0, 13628160, 0, *((generic64_t *) _var_51), *((generic64_t *) _var_52), *((generic64_t *) _var_53), *((generic32_t *) _var_54), 274877906944, 127, 2147549185, 0, *((generic32_t *) _var_55), *((generic64_t *) _var_56), *((generic32_t *) _var_57), &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
    _var_48 = &_var_8;
    _var_49 = &_var_6;
    _var_50 = &_var_5;
    _var_51 = &_var_10;
    _var_52 = &_var_14;
    _var_53 = &_var_17;
    _var_54 = &_var_18;
    _var_55 = &_var_19;
    _var_56 = &_var_21;
    _var_57 = &_var_22;
  }
}

_ABI(SystemV_x86_64)
const int8_t *dummy__(const int8_t *msg, const unreserved___locale_map *lm) {
  return msg;
}

_ABI(SystemV_x86_64)
const int8_t *unreserved___lctrans(const int8_t *msg, const unreserved___locale_map *lm) {
  return msg;
}

_ABI(SystemV_x86_64)
const int8_t *unreserved___lctrans_cur(const int8_t *msg) {
  return msg;
}

_ABI(SystemV_x86_64)
int32_t unreserved___fpclassifyl(float128_t x) {
  generic32_t _var_0;
  if (!((*((generic64_t *) (__init_local_sp() + 16)) & 0x7FFF) | ((uint64_t) *((generic64_t *) (__init_local_sp() + 8)) >> 63))) {
    generic32_t _var_1;
    _var_1 = !*((generic64_t *) (__init_local_sp() + 8)) ? 2 : 3;
    _var_0 = _var_1;
  } else {
    _var_0 = 0;
    if (!((int64_t) *((generic64_t *) (__init_local_sp() + 8)) > -1)) {
      _var_0 = 4;
      if ((*((generic64_t *) (__init_local_sp() + 16)) & 0x7FFF) == 32767) {
        _var_0 = !(*((generic64_t *) (__init_local_sp() + 8)) & 0x7FFFFFFFFFFFFFFF);
      }
    }
  }
  return (int32_t) _var_0;
}

_ABI(SystemV_x86_64)
int32_t unreserved___signbitl(float128_t x) {
  return (int32_t) (((uint32_t) *((generic32_t *) (__init_local_sp() + 16)) >> 15) & 0x1);
}

_ABI(SystemV_x86_64)
float128_t frexpl(float128_t x, int32_t *e) {
  struct _PACKED struct_570 {
    generic64_t offset_0;
    generic64_t offset_8;
    uint8_t padding_at_16[8];
    generic16_t offset_24;
    uint8_t padding_at_26[14];
  } _stack;
  generic32_t _var_0;
  generic8_t _var_1;
  generic8_t _var_2;
  generic8_t _var_3;
  generic8_t _var_4;
  generic8_t _var_5;
  generic8_t _var_6;
  generic8_t _var_7;
  generic8_t _var_8;
  generic32_t _var_9;
  generic8_t _var_10;
  generic8_t _var_11;
  generic8_t _var_12;
  generic8_t _var_13;
  generic8_t _var_14;
  generic8_t _var_15;
  generic8_t _var_16;
  generic8_t _var_17;
  generic32_t _var_18;
  generic8_t _var_19;
  generic8_t _var_20;
  generic8_t _var_21;
  generic8_t _var_22;
  generic8_t _var_23;
  generic8_t _var_24;
  generic8_t _var_25;
  generic8_t _var_26;
  generic64_t _var_27;
  generic16_t _var_28;
  generic64_t _var_29;
  generic16_t _var_30;
  generic64_t _var_31;
  generic16_t _var_32;
  generic64_t _var_33;
  generic16_t _var_34;
  generic64_t _var_35;
  generic16_t _var_36;
  generic64_t _var_37;
  generic16_t _var_38;
  generic64_t _var_39;
  generic16_t _var_40;
  generic64_t _var_41;
  generic16_t _var_42;
  generic32_t _var_43;
  generic8_t _var_44;
  generic8_t _var_45;
  generic8_t _var_46;
  generic8_t _var_47;
  generic8_t _var_48;
  generic8_t _var_49;
  generic8_t _var_50;
  generic8_t _var_51;
  generic64_t _var_52;
  generic16_t _var_53;
  generic64_t _var_54;
  generic16_t _var_55;
  generic64_t _var_56;
  generic16_t _var_57;
  generic64_t _var_58;
  generic16_t _var_59;
  generic64_t _var_60;
  generic16_t _var_61;
  generic64_t _var_62;
  generic16_t _var_63;
  generic64_t _var_64;
  generic16_t _var_65;
  generic64_t _var_66;
  generic16_t _var_67;
  generic32_t _var_68;
  generic8_t _var_69;
  generic8_t _var_70;
  generic8_t _var_71;
  generic8_t _var_72;
  generic8_t _var_73;
  generic8_t _var_74;
  generic8_t _var_75;
  generic8_t _var_76;
  generic64_t _var_77;
  generic16_t _var_78;
  generic64_t _var_79;
  generic16_t _var_80;
  generic64_t _var_81;
  generic16_t _var_82;
  generic64_t _var_83;
  generic16_t _var_84;
  generic64_t _var_85;
  generic16_t _var_86;
  generic64_t _var_87;
  generic16_t _var_88;
  generic64_t _var_89;
  generic16_t _var_90;
  generic64_t _var_91;
  generic16_t _var_92;
  generic8_t _var_93;
  generic8_t _var_94;
  generic64_t _var_95;
  generic16_t _var_96;
  generic32_t _var_97;
  generic8_t _var_98;
  generic8_t _var_99;
  generic8_t _var_100;
  generic8_t _var_101;
  generic8_t _var_102;
  generic8_t _var_103;
  generic8_t _var_104;
  generic8_t _var_105;
  generic64_t _var_106;
  generic16_t _var_107;
  generic64_t _var_108;
  generic16_t _var_109;
  generic64_t _var_110;
  generic16_t _var_111;
  generic64_t _var_112;
  generic16_t _var_113;
  generic64_t _var_114;
  generic16_t _var_115;
  generic64_t _var_116;
  generic16_t _var_117;
  generic64_t _var_118;
  generic16_t _var_119;
  generic64_t _var_120;
  generic16_t _var_121;
  generic32_t _var_122;
  generic8_t _var_123;
  generic8_t _var_124;
  generic8_t _var_125;
  generic8_t _var_126;
  generic8_t _var_127;
  generic8_t _var_128;
  generic8_t _var_129;
  generic8_t _var_130;
  generic64_t _var_131;
  generic16_t _var_132;
  generic64_t _var_133;
  generic16_t _var_134;
  generic64_t _var_135;
  generic16_t _var_136;
  generic64_t _var_137;
  generic16_t _var_138;
  generic64_t _var_139;
  generic16_t _var_140;
  generic64_t _var_141;
  generic16_t _var_142;
  generic64_t _var_143;
  generic16_t _var_144;
  generic64_t _var_145;
  generic16_t _var_146;
  generic32_t _var_147;
  generic8_t _var_148;
  generic8_t _var_149;
  generic8_t _var_150;
  generic8_t _var_151;
  generic8_t _var_152;
  generic8_t _var_153;
  generic8_t _var_154;
  generic8_t _var_155;
  generic64_t _var_156;
  generic16_t _var_157;
  generic64_t _var_158;
  generic16_t _var_159;
  generic64_t _var_160;
  generic16_t _var_161;
  generic64_t _var_162;
  generic16_t _var_163;
  generic64_t _var_164;
  generic16_t _var_165;
  generic64_t _var_166;
  generic16_t _var_167;
  generic64_t _var_168;
  generic16_t _var_169;
  generic64_t _var_170;
  generic16_t _var_171;
  generic32_t _var_172;
  generic8_t _var_173;
  generic8_t _var_174;
  generic8_t _var_175;
  generic8_t _var_176;
  generic8_t _var_177;
  generic8_t _var_178;
  generic8_t _var_179;
  generic8_t _var_180;
  generic64_t _var_181;
  generic8_t _var_182;
  generic64_t _var_183;
  generic16_t _var_184;
  generic64_t _var_185;
  generic16_t _var_186;
  generic64_t _var_187;
  generic16_t _var_188;
  generic64_t _var_189;
  generic16_t _var_190;
  generic64_t _var_191;
  generic16_t _var_192;
  generic64_t _var_193;
  generic16_t _var_194;
  generic64_t _var_195;
  generic16_t _var_196;
  generic64_t _var_197;
  generic16_t _var_198;
  generic64_t _var_199;
  generic16_t _var_200;
  generic64_t _var_201;
  generic16_t _var_202;
  generic64_t _var_203;
  generic16_t _var_204;
  generic64_t _var_205;
  generic16_t _var_206;
  generic64_t _var_207;
  generic16_t _var_208;
  generic64_t _var_209;
  generic16_t _var_210;
  generic64_t _var_211;
  generic16_t _var_212;
  generic64_t _var_213;
  generic16_t _var_214;
  generic64_t _var_215;
  generic16_t _var_216;
  generic32_t _var_217;
  generic8_t _var_218;
  generic8_t _var_219;
  generic8_t _var_220;
  generic8_t _var_221;
  generic8_t _var_222;
  generic8_t _var_223;
  generic8_t _var_224;
  generic8_t _var_225;
  generic32_t _var_226;
  generic8_t _var_227;
  generic8_t _var_228;
  generic8_t _var_229;
  generic8_t _var_230;
  generic8_t _var_231;
  generic8_t _var_232;
  generic8_t _var_233;
  generic8_t _var_234;
  generic64_t _var_235;
  generic16_t _var_236;
  generic64_t _var_237;
  generic16_t _var_238;
  generic64_t _var_239;
  generic16_t _var_240;
  generic64_t _var_241;
  generic16_t _var_242;
  generic64_t _var_243;
  generic16_t _var_244;
  generic64_t _var_245;
  generic16_t _var_246;
  generic64_t _var_247;
  generic16_t _var_248;
  generic64_t _var_249;
  generic16_t _var_250;
  generic32_t _var_251;
  generic8_t _var_252;
  generic8_t _var_253;
  generic8_t _var_254;
  generic8_t _var_255;
  generic8_t _var_256;
  generic8_t _var_257;
  generic8_t _var_258;
  generic8_t _var_259;
  generic32_t _var_260;
  generic8_t _var_261;
  generic8_t _var_262;
  generic8_t _var_263;
  generic8_t _var_264;
  generic8_t _var_265;
  generic8_t _var_266;
  generic8_t _var_267;
  generic8_t _var_268;
  generic64_t _var_269;
  generic16_t _var_270;
  generic64_t _var_271;
  generic16_t _var_272;
  generic64_t _var_273;
  generic16_t _var_274;
  generic64_t _var_275;
  generic16_t _var_276;
  generic64_t _var_277;
  generic16_t _var_278;
  generic64_t _var_279;
  generic16_t _var_280;
  generic64_t _var_281;
  generic16_t _var_282;
  generic64_t _var_283;
  generic16_t _var_284;
  generic64_t _var_285;
  generic32_t _var_286;
  generic64_t _var_287;
  _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 48, 0, &_var_260, &_var_261, &_var_262, &_var_263, &_var_264, &_var_265, &_var_266, &_var_267, &_var_268, &_var_269, &_var_270, &_var_271, &_var_272, &_var_273, &_var_274, &_var_275, &_var_276, &_var_277, &_var_278, &_var_279, &_var_280, &_var_281, &_var_282, &_var_283, &_var_284);
  _helper_fpush_wrapper(NULL, _var_260, &_var_251, &_var_252, &_var_253, &_var_254, &_var_255, &_var_256, &_var_257, &_var_258, &_var_259);
  _helper_fmov_ST0_STN_wrapper(NULL, 1, _var_251, _var_269, _var_270, _var_271, _var_272, _var_273, _var_274, _var_275, _var_276, _var_277, _var_278, _var_279, _var_280, _var_281, _var_282, _var_283, _var_284, &_var_235, &_var_236, &_var_237, &_var_238, &_var_239, &_var_240, &_var_241, &_var_242, &_var_243, &_var_244, &_var_245, &_var_246, &_var_247, &_var_248, &_var_249, &_var_250);
  _helper_fstt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 16, _var_251, _var_235, _var_236, _var_237, _var_238, _var_239, _var_240, _var_241, _var_242, _var_243, _var_244, _var_245, _var_246, _var_247, _var_248, _var_249, _var_250);
  _helper_fpop_wrapper(NULL, _var_251, &_var_226, &_var_227, &_var_228, &_var_229, &_var_230, &_var_231, &_var_232, &_var_233, &_var_234);
  if (!(*((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8)) & 0x7FFF)) {
    _helper_fpush_wrapper(NULL, _var_226, &_var_217, &_var_218, &_var_219, &_var_220, &_var_221, &_var_222, &_var_223, &_var_224, &_var_225);
    _helper_fldz_ST0_wrapper(NULL, _var_217, &_var_201, &_var_202, &_var_203, &_var_204, &_var_205, &_var_206, &_var_207, &_var_208, &_var_209, &_var_210, &_var_211, &_var_212, &_var_213, &_var_214, &_var_215, &_var_216);
    _helper_fxchg_ST0_STN_wrapper(NULL, 1, _var_217, _var_201, _var_202, _var_203, _var_204, _var_205, _var_206, _var_207, _var_208, _var_209, _var_210, _var_211, _var_212, _var_213, _var_214, _var_215, _var_216, &_var_185, &_var_186, &_var_187, &_var_188, &_var_189, &_var_190, &_var_191, &_var_192, &_var_193, &_var_194, &_var_195, &_var_196, &_var_197, &_var_198, &_var_199, &_var_200);
    _helper_fmov_FT0_STN_wrapper(NULL, 1, _var_217, _var_185, _var_186, _var_187, _var_188, _var_189, _var_190, _var_191, _var_192, _var_193, _var_194, _var_195, _var_196, _var_197, _var_198, _var_199, _var_200, &_var_183, &_var_184);
    _helper_fucomi_ST0_FT0_wrapper(NULL, 0, 23, 16, 0, _var_217, _var_185, _var_186, _var_187, _var_188, _var_189, _var_190, _var_191, _var_192, _var_193, _var_194, _var_195, _var_196, _var_197, _var_198, _var_199, _var_200, '\000', _var_183, _var_184, &_var_181, &_var_182);
    _helper_fpop_wrapper(NULL, _var_217, &_var_172, &_var_173, &_var_174, &_var_175, &_var_176, &_var_177, &_var_178, &_var_179, &_var_180);
    _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_172, _var_185, _var_186, _var_187, _var_188, _var_189, _var_190, _var_191, _var_192, _var_193, _var_194, _var_195, _var_196, _var_197, _var_198, _var_199, _var_200, &_var_156, &_var_157, &_var_158, &_var_159, &_var_160, &_var_161, &_var_162, &_var_163, &_var_164, &_var_165, &_var_166, &_var_167, &_var_168, &_var_169, &_var_170, &_var_171);
    _helper_fpop_wrapper(NULL, _var_172, &_var_147, &_var_148, &_var_149, &_var_150, &_var_151, &_var_152, &_var_153, &_var_154, &_var_155);
    if ((_var_181 & 0x44) == 64) {
      *e = 0;
      _var_286 = _var_147;
      _var_287 = *((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8));
    } else {
      float128_t _var_288;
      _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 48, _var_147, &_var_97, &_var_98, &_var_99, &_var_100, &_var_101, &_var_102, &_var_103, &_var_104, &_var_105, &_var_106, &_var_107, &_var_108, &_var_109, &_var_110, &_var_111, &_var_112, &_var_113, &_var_114, &_var_115, &_var_116, &_var_117, &_var_118, &_var_119, &_var_120, &_var_121);
      _helper_flds_FT0_wrapper(NULL, *((generic32_t *) ""), _var_182, '\000', '\000', &_var_94, &_var_95, &_var_96);
      _helper_fmul_ST0_FT0_wrapper(NULL, _var_97, _var_106, _var_107, _var_108, _var_109, _var_110, _var_111, _var_112, _var_113, _var_114, _var_115, _var_116, _var_117, _var_118, _var_119, _var_120, _var_121, '\000', '\000', _var_94, 'P', '\000', '\000', _var_95, _var_96, &_var_77, &_var_78, &_var_79, &_var_80, &_var_81, &_var_82, &_var_83, &_var_84, &_var_85, &_var_86, &_var_87, &_var_88, &_var_89, &_var_90, &_var_91, &_var_92, &_var_93);
      _stack.offset_8 = *((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8)) & 0xFFFF7FFF;
      _stack.offset_0 = *((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8)) & 0xFFFF7FFF;
      _helper_fstt_ST0_wrapper(NULL, &_stack, _var_97, _var_77, _var_78, _var_79, _var_80, _var_81, _var_82, _var_83, _var_84, _var_85, _var_86, _var_87, _var_88, _var_89, _var_90, _var_91, _var_92);
      _helper_fpop_wrapper(NULL, _var_97, &_var_68, &_var_69, &_var_70, &_var_71, &_var_72, &_var_73, &_var_74, &_var_75, &_var_76);
      _var_288 = frexpl((float128_t) ((number128_t) x & ((uint128_t) 0xFFFFFFFFFFFFFFFF)), e);
      _var_287 = (number64_t) _var_288;
      _var_285 = (number64_t) ((uint128_t) _var_288 >> 64);
      *e = (pointer_or_number32_t) *e - 120;
      _helper_fstt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 48, _var_68, _var_77, _var_78, _var_79, _var_80, _var_81, _var_82, _var_83, _var_84, _var_85, _var_86, _var_87, _var_88, _var_89, _var_90, _var_91, _var_92);
      _helper_fpop_wrapper(NULL, _var_68, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8);
      _var_286 = _var_0;
    }
  } else {
    _helper_fmov_STN_ST0_wrapper(NULL, 0, _var_226, _var_235, _var_236, _var_237, _var_238, _var_239, _var_240, _var_241, _var_242, _var_243, _var_244, _var_245, _var_246, _var_247, _var_248, _var_249, _var_250, &_var_131, &_var_132, &_var_133, &_var_134, &_var_135, &_var_136, &_var_137, &_var_138, &_var_139, &_var_140, &_var_141, &_var_142, &_var_143, &_var_144, &_var_145, &_var_146);
    _helper_fpop_wrapper(NULL, _var_226, &_var_122, &_var_123, &_var_124, &_var_125, &_var_126, &_var_127, &_var_128, &_var_129, &_var_130);
    _var_286 = _var_122;
    _var_285 = *((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8)) & 0x7FFF;
    _var_287 = *((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8));
    if ((*((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8)) & 0x7FFF) != 32767) {
      _var_285 = (number32_t) (*((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8)) & 0x7FFF) - 16382;
      _var_287 = (*((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8)) & 0xFFFFFFFFFFFF8000) | 0x3FFE;
      *e = (number32_t) (*((generic64_t *) ((pointer_or_number64_t) &(&_stack)[1].offset_8 + 8)) & 0x7FFF) - 16382;
      _stack.offset_24 = (number16_t) _var_287;
      _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 16, _var_122, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32, &_var_33, &_var_34, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42);
      _helper_fstt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 48, _var_18, _var_27, _var_28, _var_29, _var_30, _var_31, _var_32, _var_33, _var_34, _var_35, _var_36, _var_37, _var_38, _var_39, _var_40, _var_41, _var_42);
      _helper_fpop_wrapper(NULL, _var_18, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17);
      _var_286 = _var_9;
    }
  }
  _helper_fldt_ST0_wrapper(NULL, (pointer_or_number64_t) &_stack + 48, _var_286, &_var_43, &_var_44, &_var_45, &_var_46, &_var_47, &_var_48, &_var_49, &_var_50, &_var_51, &_var_52, &_var_53, &_var_54, &_var_55, &_var_56, &_var_57, &_var_58, &_var_59, &_var_60, &_var_61, &_var_62, &_var_63, &_var_64, &_var_65, &_var_66, &_var_67);
  return (float128_t) (((number128_t) _var_285 << 64) | _var_287);
}

_ABI(SystemV_x86_64)
int32_t wctomb(int8_t *s, wchar_t wc) {
  generic32_t _var_0;
  _var_0 = 0;
  if (s) {
    size_t _var_1;
    _var_1 = wcrtomb(s, wc, (typedef_350) NULL);
    _var_0 = (number32_t) _var_1;
  }
  return (int32_t) _var_0;
}

_ABI(SystemV_x86_64)
int32_t unreserved___lockfile(FILE_ *f) {
  generic32_t _var_0;
  _var_0 = 0;
  if ((pointer_or_number32_t) f->lock != *((generic32_t *) (*((generic64_t *) NULL) + 56))) {
    while (true) {
      generic64_t _var_1;
      _helper_lock();
      if (!f->lock) {
        f->lock = *((generic32_t *) (*((generic64_t *) NULL) + 56));
        _var_1 = 0;
      } else {
        _var_1 = (uint64_t) f->lock;
      }
      _helper_unlock();
      if (!_var_1) {
        break;
      }
      unreserved___wait(&f->lock, &f->waiters, (int32_t) (number32_t) _var_1, (int32_t) 1);
    }
    _var_0 = 1;
  }
  return (int32_t) _var_0;
}

_ABI(SystemV_x86_64)
void unreserved___unlockfile(FILE_ *f) {
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  generic32_t _var_24;
  generic64_t _var_25;
  generic64_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic32_t _var_29;
  generic64_t _var_30;
  generic32_t _var_31;
  generic32_t _var_32;
  generic32_t _var_33;
  generic64_t _var_34;
  generic32_t _var_35;
  generic32_t _var_36;
  generic32_t _var_37;
  generic64_t _var_38;
  generic32_t _var_39;
  generic32_t _var_40;
  generic64_t _var_41;
  generic32_t _var_42;
  generic32_t _var_43;
  generic32_t _var_44;
  generic64_t _var_45;
  generic32_t _var_46;
  generic8_t _var_47;
  f->lock = 0;
  _helper_lock();
  _helper_unlock();
  if (f->waiters) {
    _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10830, _undef_generic64_t(), 202, _undef_generic64_t(), 202, _undef_generic64_t(), _undef_generic64_t(), f, (pointer_or_number64_t) f + 140, 1, 129, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32, &_var_33, &_var_34, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42, &_var_43, &_var_44, &_var_45, &_var_46, &_var_47);
    if (_var_27 == (pointer_or_number64_t) -38) {
      _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10844, _undef_generic64_t(), 202, _undef_generic64_t(), 202, _undef_generic64_t(), _undef_generic64_t(), f, (pointer_or_number64_t) f + 140, 1, 1, _var_29, _var_30, _var_32, 0, 0, 15727360, 0, 13628160, 0, _var_34, _var_38, _var_41, _var_42, 274877906944, 127, 2147549185, 0, _var_43, _var_45, _var_46, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
    }
  }
}

_ABI(SystemV_x86_64)
int32_t dummy___(int32_t fd) {
  return fd;
}

_ABI(SystemV_x86_64)
int32_t unreserved___stdio_close(FILE_ *f) {
  int64_t _var_0;
  int32_t _var_1;
  generic32_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic64_t _var_8;
  generic32_t _var_9;
  generic32_t _var_10;
  generic32_t _var_11;
  generic64_t _var_12;
  generic32_t _var_13;
  generic32_t _var_14;
  generic32_t _var_15;
  generic64_t _var_16;
  generic32_t _var_17;
  generic32_t _var_18;
  generic64_t _var_19;
  generic32_t _var_20;
  generic32_t _var_21;
  generic32_t _var_22;
  generic64_t _var_23;
  generic32_t _var_24;
  generic8_t _var_25;
  _var_1 = dummy___(f->fd);
  _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10867, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 3, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), (int64_t) _var_1, _undef_generic64_t(), _undef_generic64_t(), 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23, &_var_24, &_var_25);
  _var_0 = unreserved___syscall_ret(_var_5);
  return (int32_t) (number32_t) _var_0;
}

_ABI(SystemV_x86_64)
off_t unreserved___stdio_seek(FILE_ *f, off_t off, int32_t whence) {
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  generic64_t _var_24;
  _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10891, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 8, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), (int64_t) f->fd, (int64_t) whence, off, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
  _var_24 = _var_3;
  if (_var_24 > (uint64_t) -4096) {
    int32_t *_var_25;
    *((generic64_t *) (_var_4 - 16)) = _var_3;
    _var_25 = unreserved___errno_location();
    *_var_25 = 0 - (number32_t) *((generic64_t *) (_var_4 - 16));
    _var_24 = 18446744073709551615U;
  }
  return (off_t) _var_24;
}

_ABI(SystemV_x86_64)
size_t unreserved___stdout_write(FILE_ *f, const uint8_t *buf, size_t len) {
  struct _PACKED struct_579 {
    uint8_t padding_at_0[1];
  } _stack;
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  f->write = unreserved___stdio_write;
  if (!(*((generic8_t *) f) & 0x40)) {
    _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10946, len, f, buf, 16, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), (int64_t) f->fd, (pointer_or_number64_t) &_stack - 15, 21523, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
    if (_var_3) {
      f->lbf = '\377';
    }
  }
  size_t _var_24;
  _var_24 = unreserved___stdio_write(f, buf, len);
  return _var_24;
}

_ABI(SystemV_x86_64)
size_t unreserved___fwritex(typedef_312 s, size_t l, typedef_303 f) {
  generic64_t _var_0;
  if (!f->wend) {
    int32_t _var_1;
    _var_1 = unreserved___towrite(f);
    _var_0 = 0;
    if (_var_1) {
      return _var_0;
    }
  }
  if ((pointer_or_number64_t) f->wend - (number64_t) f->wpos < l) {
    _var_0 = f->write;
  } else {
    generic64_t _var_2;
    generic64_t _var_3;
    _var_2 = s;
    _var_3 = l;
    if (!(f->lbf < (int8_t) 0 || !l)) {
      generic64_t _var_4;
      generic64_t _var_5;
      _var_4 = 0;
      _var_5 = l;
      while (true) {
        if ((pointer_or_number8_t) s[~_var_4 + l] == '\n') {
          pointer_or_number64_t _var_6;
          pointer_or_number64_t _var_7;
          artificial_struct_returned_by_rawfunction_25 _var_8;
          _var_8 = ((rawfunction_25 *) f->write)(_undef_generic64_t(), _var_5, (pointer_or_number64_t) s, (pointer_or_number64_t) f, _undef_generic64_t(), _undef_generic64_t());
          _var_7 = _var_8.register_rax;
          _var_6 = _var_8.register_rdx;
          _var_0 = _var_5;
          if (_var_5 > _var_7) {
            return _var_0;
          }
          _var_3 = l - _var_5;
          _var_2 = (pointer_or_number64_t) &s[l] - _var_4;
        } else {
          generic8_t _var_9;
          _var_5 = _var_5 - 1;
          _var_9 = ~_var_4 == 0 - l;
          _var_4 = _var_4 + 1;
          if (!(_var_9)) {
            continue;
          }
          _var_2 = s;
          _var_3 = l;
        }
        break;
      }
    }
    struct_718 *_var_10;
    _var_10 = memcpy((struct_718 *) f->wpos, (union_596 *) _var_2, _var_3);
    f->wpos = &f->wpos[_var_3];
    _var_0 = l;
  }
  return _var_0;
}

_ABI(SystemV_x86_64)
size_t fwrite_unlocked(typedef_314 src, size_t size, size_t nmemb, typedef_303 f) {
  struct _PACKED struct_582 {
    uint8_t padding_at_0[16];
    union_725 *offset_16;
    uint8_t padding_at_24[32];
  } _stack;
  generic8_t _var_0;
  _stack.offset_16 = f;
  _var_0 = true;
  if (f->lock > -1) {
    int32_t _var_1;
    _var_1 = unreserved___lockfile(f);
    _var_0 = !_var_1;
  }
  size_t _var_2;
  _var_2 = unreserved___fwritex((typedef_312) src, nmemb * size, f);
  if (!(_var_0)) {
    unreserved___unlockfile(f);
  }
  generic64_t _var_3;
  _var_3 = nmemb;
  if (nmemb * size != _var_2) {
    _var_3 = _var_2 / size;
  }
  return _var_3;
}

_ABI(SystemV_x86_64)
struct_718 *memcpy(struct_718 *argument_0, union_596 *argument_1, generic64_t argument_2) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  _var_0 = argument_0;
  _var_1 = argument_2;
  _var_2 = argument_1;
  if (!(argument_2 < 8 || !((number64_t) argument_0 & 0x7))) {
    generic64_t _var_3;
    generic64_t _var_4;
    generic64_t _var_5;
    generic64_t _var_6;
    _var_3 = 0;
    _var_4 = argument_0;
    _var_5 = argument_2;
    _var_6 = argument_1;
    generic64_t _var_7;
    generic64_t _var_8;
    generic8_t _var_9;
    do {
      _var_8 = _var_4;
      _var_7 = _var_6;
      ((struct_718 *) _var_8)->offset_0.member_1 = ((union_596 *) _var_7)->member_1;
      _var_5 = _var_5 - 1;
      _var_9 = !(((pointer_or_number64_t) &argument_0->offset_0.member_2.offset_1 + _var_3 * 1) & 0x7);
      _var_3 = _var_3 + 1;
      _var_4 = &((struct_718 *) _var_8)->offset_0.member_2.offset_1;
      _var_6 = &((union_596 *) _var_7)->member_0.offset_1;
    } while (!(_var_9));
    _var_0 = &((struct_718 *) _var_8)->offset_0.member_2.offset_1;
    _var_2 = &((union_596 *) _var_7)->member_0.offset_1;
    _var_1 = _var_5;
  }
  generic64_t _var_10;
  generic64_t _var_11;
  _var_10 = _var_0;
  _var_11 = _var_2;
  if (!(_var_1 < 8)) {
    generic64_t _var_12;
    generic64_t _var_13;
    generic64_t _var_14;
    _var_12 = 0;
    _var_13 = _var_2;
    _var_14 = _var_0;
    do {
      _var_12 = _var_12 + 1;
      ((struct_718 *) _var_14)->offset_0.member_0 = ((union_596 *) _var_13)->member_3;
      _var_13 = &((union_596 *) _var_13)->member_5.offset_8;
      _var_14 = &((struct_718 *) _var_14)->offset_8;
    } while (_var_1 >> 3 != _var_12);
    _var_10 = _var_0 + (_var_1 & 0xFFFFFFFFFFFFFFF8);
    _var_11 = _var_2 + (_var_1 & 0xFFFFFFFFFFFFFFF8);
  }
  if ((_var_1 & 0x7)) {
    generic64_t _var_15;
    generic64_t _var_16;
    generic64_t _var_17;
    _var_15 = _var_10;
    _var_16 = _var_1 & 0x7;
    _var_17 = _var_11;
    do {
      ((struct_718 *) _var_15)->offset_0.member_1 = ((union_596 *) _var_17)->member_1;
      _var_16 = (_var_16 - 1) & 0xFFFFFFFF;
      _var_15 = &((struct_718 *) _var_15)->offset_0.member_2.offset_1;
      _var_17 = &((union_596 *) _var_17)->member_0.offset_1;
    } while (_var_16);
  }
  return argument_0;
}

_ABI(SystemV_x86_64)
generic64_t unreserved___set_thread_area(struct_703 *argument_0) {
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 11342, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 158, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 4098, _undef_generic64_t(), argument_0, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
  return _var_3;
}

_ABI(SystemV_x86_64)
void unreserved___wait(typedef_315 *addr, typedef_315 *waiters, int32_t val, int32_t priv) {
  struct _PACKED struct_576 {
    generic64_t offset_0;
    generic64_t offset_8;
  } _stack;
  generic64_t _var_0;
  generic32_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic32_t _var_6;
  generic64_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic32_t _var_10;
  generic64_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic32_t _var_14;
  generic64_t _var_15;
  generic32_t _var_16;
  generic32_t _var_17;
  generic64_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic32_t _var_21;
  generic64_t _var_22;
  generic32_t _var_23;
  generic8_t _var_24;
  generic32_t _var_25;
  generic64_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic64_t _var_29;
  generic32_t _var_30;
  generic64_t _var_31;
  generic32_t _var_32;
  generic32_t _var_33;
  generic32_t _var_34;
  generic64_t _var_35;
  generic32_t _var_36;
  generic32_t _var_37;
  generic32_t _var_38;
  generic64_t _var_39;
  generic32_t _var_40;
  generic32_t _var_41;
  generic64_t _var_42;
  generic32_t _var_43;
  generic32_t _var_44;
  generic32_t _var_45;
  generic64_t _var_46;
  generic32_t _var_47;
  generic8_t _var_48;
  generic32_t _var_49;
  generic64_t _var_50;
  generic64_t _var_51;
  generic64_t _var_52;
  generic64_t _var_53;
  generic32_t _var_54;
  generic64_t _var_55;
  generic32_t _var_56;
  generic32_t _var_57;
  generic32_t _var_58;
  generic64_t _var_59;
  generic32_t _var_60;
  generic32_t _var_61;
  generic32_t _var_62;
  generic64_t _var_63;
  generic32_t _var_64;
  generic32_t _var_65;
  generic64_t _var_66;
  generic32_t _var_67;
  generic32_t _var_68;
  generic32_t _var_69;
  generic64_t _var_70;
  generic32_t _var_71;
  generic8_t _var_72;
  _stack.offset_8 = 202;
  _stack.offset_0 = 202;
  _var_0 = !priv ? 0 : 128;
  if ((waiters) && (*waiters)) {
    _helper_lock();
    *waiters = (pointer_or_number32_t) *waiters + 1;
    _helper_unlock();
    if ((pointer_or_number32_t) *addr == (pointer_or_number32_t) val) {
      generic32_t _var_73;
      generic64_t _var_74;
      generic32_t _var_75;
      generic32_t _var_76;
      generic64_t _var_77;
      generic64_t _var_78;
      generic64_t _var_79;
      generic32_t _var_80;
      generic64_t _var_81;
      generic32_t _var_82;
      _var_73 = 4294967295;
      _var_74 = 0;
      _var_75 = 0;
      _var_76 = 65535;
      _var_77 = 0;
      _var_78 = 0;
      _var_79 = 0;
      _var_80 = 4243635;
      _var_81 = 514;
      _var_82 = 4294967295;
      do {
        generic32_t _var_83;
        generic64_t _var_84;
        generic32_t _var_85;
        generic64_t _var_86;
        generic64_t _var_87;
        generic64_t _var_88;
        generic32_t _var_89;
        generic32_t _var_90;
        generic64_t _var_91;
        generic32_t _var_92;
        _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 11437, 0, waiters, _var_0, 202, 202, (uint64_t) val, _var_0, addr, (int64_t) val, _var_0, _var_82, _var_81, _var_80, 0, 0, 15727360, 0, 13628160, 0, _var_79, _var_78, _var_77, _var_76, 274877906944, 127, 2147549185, 0, _var_75, _var_74, _var_73, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32, &_var_33, &_var_34, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42, &_var_43, &_var_44, &_var_45, &_var_46, &_var_47, &_var_48);
        _var_83 = _var_30;
        _var_84 = _var_31;
        _var_85 = _var_33;
        _var_86 = _var_35;
        _var_87 = _var_39;
        _var_88 = _var_42;
        _var_89 = _var_43;
        _var_90 = _var_44;
        _var_91 = _var_46;
        _var_92 = _var_47;
        if (_var_28 == (pointer_or_number64_t) -38) {
          _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 11450, 0, waiters, _var_0, 202, 202, (uint64_t) val, _var_0, addr, (int64_t) val, 0, _var_30, _var_31, _var_33, 0, 0, 15727360, 0, 13628160, 0, _var_35, _var_39, _var_42, _var_43, 274877906944, 127, 2147549185, 0, _var_44, _var_46, _var_47, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23, &_var_24);
          _var_83 = _var_6;
          _var_84 = _var_7;
          _var_85 = _var_9;
          _var_86 = _var_11;
          _var_87 = _var_15;
          _var_88 = _var_18;
          _var_89 = _var_19;
          _var_90 = _var_20;
          _var_91 = _var_22;
          _var_92 = _var_23;
        }
      } while ((pointer_or_number32_t) *addr == (pointer_or_number32_t) val);
    }
    _helper_lock();
    *waiters = (pointer_or_number32_t) *waiters - 1;
    _helper_unlock();
    return;
  }
  if ((pointer_or_number32_t) *addr != (pointer_or_number32_t) val) {
    return;
  }
  _helper_pause_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 11383, 0, waiters, _undef_generic64_t(), 100, 202, (uint64_t) val, _var_0, addr, (uint64_t) *addr, waiters, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_49, &_var_50, &_var_51, &_var_52, &_var_53, &_var_54, &_var_55, &_var_56, &_var_57, &_var_58, &_var_59, &_var_60, &_var_61, &_var_62, &_var_63, &_var_64, &_var_65, &_var_66, &_var_67, &_var_68, &_var_69, &_var_70, &_var_71, &_var_72);
  __abort("A longjmp was taken");
}

_ABI(SystemV_x86_64)
int64_t unreserved___syscall_ret(uint64_t r) {
  struct _PACKED struct_578 {
    uint8_t padding_at_0[8];
    generic64_t offset_8;
    uint8_t padding_at_16[8];
  } _stack;
  generic64_t _var_0;
  _var_0 = r;
  if (r > (uint64_t) -4096) {
    int32_t *_var_1;
    _stack.offset_8 = r;
    _var_1 = unreserved___errno_location();
    *_var_1 = 0 - (number32_t) _stack.offset_8;
    _var_0 = 18446744073709551615U;
  }
  return (int64_t) _var_0;
}

_ABI(SystemV_x86_64)
size_t wcrtomb(typedef_332 s, wchar_t wc, typedef_350 st) {
  generic64_t _var_0;
  _var_0 = 1;
  if (s) {
    if ((uint32_t) wc < 128) {
      *s = (number8_t) wc;
      _var_0 = 1;
    } else {
      if (!*((generic64_t *) *((generic64_t *) (*((generic64_t *) NULL) + 256)))) {
        if (((number32_t) wc & 0xFFFFFF80) == 57216) {
          *s = (number8_t) wc;
          _var_0 = 1;
          return _var_0;
        }
      } else {
        if ((uint32_t) wc < 2048) {
          s[1] = ((number8_t) wc & 0x3F) | 0x80;
          *s = (number8_t) ((uint64_t) wc >> 6) | 0xC0;
          _var_0 = 2;
          return _var_0;
        }
        if (!((((uint64_t) wc + 4294909952) & 0xFFFFE000) != 0 && (uint32_t) wc > 55295)) {
          *s = (number8_t) ((uint64_t) wc >> 12) | 0xE0;
          s[2] = ((number8_t) wc & 0x3F) | 0x80;
          s[1] = ((number8_t) ((uint64_t) wc >> 6) & 0x3F) | 0x80;
          _var_0 = 3;
          return _var_0;
        }
        if ((uint32_t) wc < 1114112 && (uint32_t) wc > 65535) {
          *s = (number8_t) ((uint64_t) wc >> 18) | 0xF0;
          s[1] = ((number8_t) ((uint64_t) wc >> 12) & 0x3F) | 0x80;
          s[3] = ((number8_t) wc & 0x3F) | 0x80;
          s[2] = ((number8_t) ((uint64_t) wc >> 6) & 0x3F) | 0x80;
          _var_0 = 4;
          return _var_0;
        }
      }
      int32_t *_var_1;
      _var_1 = unreserved___errno_location();
      *_var_1 = 84;
      _var_0 = 18446744073709551615U;
    }
  }
  return _var_0;
}

_ABI(SystemV_x86_64)
size_t unreserved___stdio_write(FILE_ *f, const uint8_t *buf, size_t len) {
  struct _PACKED struct_580 {
    generic64_t offset_0;
    generic64_t offset_8;
    generic64_t offset_16;
    generic64_t offset_24;
    uint8_t padding_at_32[32];
    generic64_t offset_64;
    uint8_t padding_at_72[16];
  } _stack;
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  generic32_t _var_24;
  generic64_t _var_25;
  generic32_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic64_t _var_29;
  generic64_t _var_30;
  generic32_t _var_31;
  generic64_t _var_32;
  generic64_t _var_33;
  generic32_t _var_34;
  generic64_t _var_35;
  generic32_t _var_36;
  _var_27 = &_stack;
  ((struct_580 *) _var_27)->offset_64 = 20;
  ((struct_580 *) _var_27)->offset_16 = buf;
  ((struct_580 *) _var_27)->offset_0 = f->wbase;
  ((struct_580 *) _var_27)->offset_24 = len;
  ((struct_580 *) _var_27)->offset_8 = (pointer_or_number64_t) f->wpos - (number64_t) f->wbase;
  _var_33 = (pointer_or_number64_t) f->wpos - (number64_t) f->wbase + len;
  _var_24 = 4294967295;
  _var_25 = 514;
  _var_26 = 4243635;
  _var_28 = 0;
  _var_29 = 0;
  _var_30 = 0;
  _var_31 = 65535;
  _var_32 = 2;
  _var_34 = 0;
  _var_35 = 0;
  _var_36 = 4294967295;
  generic64_t _var_37;
  while (true) {
    int64_t _var_38;
    _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 11871, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 20, _var_27, f, _undef_generic64_t(), (int64_t) f->fd, (int64_t) ((number64_t) _var_32 << 32) >> 32, _var_27, _var_24, _var_25, _var_26, 0, 0, 15727360, 0, 13628160, 0, _var_28, _var_29, _var_30, _var_31, 274877906944, 127, 2147549185, 0, _var_34, _var_35, _var_36, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
    _var_38 = unreserved___syscall_ret(_var_3);
    if (_var_33 == (pointer_or_number64_t) _var_38) {
      f->wbase = f->buf;
      f->wend = &f->buf[f->buf_size];
      f->wpos = f->buf;
      _var_37 = len;
    } else {
      if (_var_38 > -1) {
        generic64_t _var_39;
        generic64_t _var_40;
        generic64_t _var_41;
        generic64_t _var_42;
        _var_39 = ((struct_580 *) _var_27)->offset_8;
        _var_33 = _var_33 - (number64_t) _var_38;
        _var_42 = (uint64_t) _var_38 > _var_39 ? (generic64_t) &((struct_580 *) _var_27)->offset_16 : _var_27;
        _var_27 = _var_42;
        _var_41 = (uint64_t) _var_38 > _var_39 ? _var_39 : 0;
        _var_40 = (uint64_t) _var_38 > _var_39 ? (_var_32 + 4294967295) & 0xFFFFFFFF : _var_32;
        _var_32 = _var_40;
        *((generic64_t *) _var_27) = *((generic64_t *) _var_27) + ((pointer_or_number64_t) _var_38 - _var_41);
        *((generic64_t *) (_var_27 + 8)) = *((generic64_t *) (_var_27 + 8)) - ((pointer_or_number64_t) _var_38 - _var_41);
        _var_24 = _var_5;
        _var_25 = _var_6;
        _var_26 = _var_8;
        _var_28 = _var_10;
        _var_29 = _var_14;
        _var_30 = _var_17;
        _var_31 = _var_18;
        _var_34 = _var_19;
        _var_35 = _var_21;
        _var_36 = _var_22;
        continue;
      }
      f->flags = f->flags | 0x20;
      f->wend = 0;
      f->wbase = 0;
      f->wpos = 0;
      _var_37 = 0;
      if ((_var_32 & 0xFFFFFFFF) != 2) {
        _var_37 = len - ((struct_580 *) _var_27)->offset_8;
      }
    }
    break;
  }
  return _var_37;
}

_ABI(SystemV_x86_64)
int32_t unreserved___towrite(FILE_ *f) {
  generic32_t _var_0;
  f->mode = ((pointer_or_number8_t) f->mode - '\001') | (number8_t) f->mode;
  if (!(f->flags & 0x8)) {
    f->rend = 0;
    f->rpos = 0;
    f->wbase = f->buf;
    f->wpos = f->buf;
    f->wend = &f->buf[f->buf_size];
    _var_0 = 0;
  } else {
    f->flags = f->flags | 0x20;
    _var_0 = 4294967295;
  }
  return (int32_t) _var_0;
}

_ABI(SystemV_x86_64)
void unreserved___towrite_needs_stdio_exit(void) {
  struct _PACKED struct_583 {
    uint8_t padding_at_0[8];
  } _stack;
  FILE_ **_var_0;
  generic32_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic32_t _var_6;
  generic64_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic32_t _var_10;
  generic64_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic32_t _var_14;
  generic64_t _var_15;
  generic32_t _var_16;
  generic32_t _var_17;
  generic64_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic32_t _var_21;
  generic64_t _var_22;
  generic32_t _var_23;
  generic8_t _var_24;
  _var_0 = unreserved___ofl_lock();
  if (*_var_0) {
    FILE_ *_var_25;
    _var_25 = *_var_0;
    do {
      close_file(_var_25);
      _var_25 = _var_25->next;
    } while (_var_25);
  }
  close_file((FILE_ *) segment_0x406fd0_Generic64_2224.unreserved__bss.dummy_file);
  if (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used) {
    if ((int32_t) *((generic32_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 140)) > -1) {
      int32_t _var_26;
      _var_26 = unreserved___lockfile((FILE_ *) segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used);
    }
    if (*((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 40)) > *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 56))) {
      pointer_or_number64_t _var_27;
      pointer_or_number64_t _var_28;
      artificial_struct_returned_by_rawfunction_25 _var_29;
      _var_29 = ((rawfunction_25 *) *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 72)))(_undef_generic64_t(), 0, 0, segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used, _undef_generic64_t(), _undef_generic64_t());
      _var_28 = _var_29.register_rax;
      _var_27 = _var_29.register_rdx;
    }
    if (*((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 8)) < *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 16))) {
      _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10891, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 8, *((generic64_t *) &_stack), *((generic64_t *) &_stack), _undef_generic64_t(), (int64_t) *((generic32_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 120)), 1, *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 8)) - *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 16)), 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23, &_var_24);
      if (_var_4 > (uint64_t) -4096) {
        int32_t *_var_30;
        *((generic64_t *) (_var_5 - 16)) = _var_4;
        _var_30 = unreserved___errno_location();
        *_var_30 = 0 - (number32_t) *((generic64_t *) (_var_5 - 16));
      }
    }
  }
}

_ABI(SystemV_x86_64)
void close_file(FILE_ *f) {
  struct _PACKED struct_585 {
    uint8_t padding_at_0[8];
  } _stack;
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  if (f) {
    if (f->lock > -1) {
      int32_t _var_24;
      _var_24 = unreserved___lockfile(f);
    }
    if ((uint64_t) f->wpos > (uint64_t) f->wbase) {
      pointer_or_number64_t _var_25;
      pointer_or_number64_t _var_26;
      artificial_struct_returned_by_rawfunction_25 _var_27;
      _var_27 = ((rawfunction_25 *) f->write)(_undef_generic64_t(), 0, 0, (pointer_or_number64_t) f, _undef_generic64_t(), _undef_generic64_t());
      _var_26 = _var_27.register_rax;
      _var_25 = _var_27.register_rdx;
    }
    if ((uint64_t) f->rpos < (uint64_t) f->rend) {
      _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10891, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 8, *((generic64_t *) &_stack), _undef_generic64_t(), _undef_generic64_t(), (int64_t) f->fd, 1, (pointer_or_number64_t) f->rpos - (number64_t) f->rend, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
      if (_var_3 > (uint64_t) -4096) {
        int32_t *_var_28;
        *((generic64_t *) (_var_4 - 16)) = _var_3;
        _var_28 = unreserved___errno_location();
        *_var_28 = 0 - (number32_t) *((generic64_t *) (_var_4 - 16));
      }
    }
  }
}

_ABI(SystemV_x86_64)
void unreserved___stdio_exit(void) {
  struct _PACKED struct_565 {
    uint8_t padding_at_0[8];
  } _stack;
  FILE_ **_var_0;
  generic32_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic32_t _var_6;
  generic64_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic32_t _var_10;
  generic64_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic32_t _var_14;
  generic64_t _var_15;
  generic32_t _var_16;
  generic32_t _var_17;
  generic64_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic32_t _var_21;
  generic64_t _var_22;
  generic32_t _var_23;
  generic8_t _var_24;
  _var_0 = unreserved___ofl_lock();
  if (*_var_0) {
    FILE_ *_var_25;
    _var_25 = *_var_0;
    do {
      close_file(_var_25);
      _var_25 = _var_25->next;
    } while (_var_25);
  }
  close_file((FILE_ *) segment_0x406fd0_Generic64_2224.unreserved__bss.dummy_file);
  if (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used) {
    if ((int32_t) *((generic32_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 140)) > -1) {
      int32_t _var_26;
      _var_26 = unreserved___lockfile((FILE_ *) segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used);
    }
    if (*((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 40)) > *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 56))) {
      pointer_or_number64_t _var_27;
      pointer_or_number64_t _var_28;
      artificial_struct_returned_by_rawfunction_25 _var_29;
      _var_29 = ((rawfunction_25 *) *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 72)))(_undef_generic64_t(), 0, 0, segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used, _undef_generic64_t(), _undef_generic64_t());
      _var_28 = _var_29.register_rax;
      _var_27 = _var_29.register_rdx;
    }
    if (*((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 8)) < *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 16))) {
      _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 10891, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), 8, *((generic64_t *) &_stack), *((generic64_t *) &_stack), _undef_generic64_t(), (int64_t) *((generic32_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 120)), 1, *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 8)) - *((generic64_t *) (segment_0x406fd0_Generic64_2224.unreserved__data.unreserved___stdout_used + 16)), 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23, &_var_24);
      if (_var_4 > (uint64_t) -4096) {
        int32_t *_var_30;
        *((generic64_t *) (_var_5 - 16)) = _var_4;
        _var_30 = unreserved___errno_location();
        *_var_30 = 0 - (number32_t) *((generic64_t *) (_var_5 - 16));
      }
    }
  }
}

_ABI(SystemV_x86_64)
FILE_ **unreserved___ofl_lock(void) {
  struct _PACKED struct_584 {
    uint8_t padding_at_0[8];
  } _stack;
  *((generic64_t **) &_stack) = &segment_0x406fd0_Generic64_2224.unreserved__bss.ofl_head;
  unreserved___lock((typedef_407 *) &segment_0x406fd0_Generic64_2224.unreserved__bss.ofl_lock);
  return (FILE_ **) &segment_0x406fd0_Generic64_2224.unreserved__bss.ofl_head;
}

_ABI(SystemV_x86_64)
void unreserved___ofl_unlock(void) {
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  generic32_t _var_24;
  generic64_t _var_25;
  generic64_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic32_t _var_29;
  generic64_t _var_30;
  generic32_t _var_31;
  generic32_t _var_32;
  generic32_t _var_33;
  generic64_t _var_34;
  generic32_t _var_35;
  generic32_t _var_36;
  generic32_t _var_37;
  generic64_t _var_38;
  generic32_t _var_39;
  generic32_t _var_40;
  generic64_t _var_41;
  generic32_t _var_42;
  generic32_t _var_43;
  generic32_t _var_44;
  generic64_t _var_45;
  generic32_t _var_46;
  generic8_t _var_47;
  if (*((generic32_t *) &segment_0x406fd0_Generic64_2224.unreserved__bss.ofl_lock)) {
    *((generic32_t *) &segment_0x406fd0_Generic64_2224.unreserved__bss.ofl_lock) = 0;
    _helper_lock();
    _helper_unlock();
    if (*((generic32_t *) ((pointer_or_number64_t) &segment_0x406fd0_Generic64_2224.unreserved__bss.ofl_lock + 4))) {
      _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 12342, _undef_generic64_t(), 202, _undef_generic64_t(), 202, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), (pointer_or_number64_t) &segment_0x406fd0_Generic64_2224 + 2032, 1, 129, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32, &_var_33, &_var_34, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42, &_var_43, &_var_44, &_var_45, &_var_46, &_var_47);
      if (_var_27 == (pointer_or_number64_t) -38) {
        _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 12356, _undef_generic64_t(), 202, _undef_generic64_t(), 202, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), (pointer_or_number64_t) &segment_0x406fd0_Generic64_2224 + 2032, 1, 1, _var_29, _var_30, _var_32, 0, 0, 15727360, 0, 13628160, 0, _var_34, _var_38, _var_41, _var_42, 274877906944, 127, 2147549185, 0, _var_43, _var_45, _var_46, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
      }
    }
  }
}

_ABI(SystemV_x86_64)
void unreserved___lock(typedef_407 *l) {
  struct _PACKED struct_586 {
    generic64_t offset_0;
    uint8_t padding_at_8[8];
    generic64_t offset_16;
  } _stack;
  if (segment_0x406fd0_Generic64_2224.unreserved__bss.unreserved___libc__.offset_12) {
    _stack.offset_16 = 1;
    _stack.offset_0 = 1;
    _helper_lock();
    *l = 1;
    _helper_unlock();
    if (*l) {
      do {
        unreserved___wait(l, &l[1], (int32_t) 1, (int32_t) 1);
        _helper_lock();
        *l = 1;
        _helper_unlock();
      } while (*l);
    }
  }
}

_ABI(SystemV_x86_64)
void unreserved___unlock(typedef_407 *l) {
  generic32_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic32_t _var_7;
  generic32_t _var_8;
  generic32_t _var_9;
  generic64_t _var_10;
  generic32_t _var_11;
  generic32_t _var_12;
  generic32_t _var_13;
  generic64_t _var_14;
  generic32_t _var_15;
  generic32_t _var_16;
  generic64_t _var_17;
  generic32_t _var_18;
  generic32_t _var_19;
  generic32_t _var_20;
  generic64_t _var_21;
  generic32_t _var_22;
  generic8_t _var_23;
  generic32_t _var_24;
  generic64_t _var_25;
  generic64_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic32_t _var_29;
  generic64_t _var_30;
  generic32_t _var_31;
  generic32_t _var_32;
  generic32_t _var_33;
  generic64_t _var_34;
  generic32_t _var_35;
  generic32_t _var_36;
  generic32_t _var_37;
  generic64_t _var_38;
  generic32_t _var_39;
  generic32_t _var_40;
  generic64_t _var_41;
  generic32_t _var_42;
  generic32_t _var_43;
  generic32_t _var_44;
  generic64_t _var_45;
  generic32_t _var_46;
  generic8_t _var_47;
  if (*l) {
    *l = 0;
    _helper_lock();
    _helper_unlock();
    if (l[1]) {
      _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 12342, _undef_generic64_t(), 202, _undef_generic64_t(), 202, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), l, 1, 129, 4294967295, 514, 4243635, 0, 0, 15727360, 0, 13628160, 0, 0, 0, 0, 65535, 274877906944, 127, 2147549185, 0, 0, 0, 4294967295, &_var_24, &_var_25, &_var_26, &_var_27, &_var_28, &_var_29, &_var_30, &_var_31, &_var_32, &_var_33, &_var_34, &_var_35, &_var_36, &_var_37, &_var_38, &_var_39, &_var_40, &_var_41, &_var_42, &_var_43, &_var_44, &_var_45, &_var_46, &_var_47);
      if (_var_27 == (pointer_or_number64_t) -38) {
        _helper_syscall_wrapper(NULL, 2, (pointer_or_number64_t) &segment_0x401000_Generic64_12441 + 12356, _undef_generic64_t(), 202, _undef_generic64_t(), 202, _undef_generic64_t(), _undef_generic64_t(), _undef_generic64_t(), l, 1, 1, _var_29, _var_30, _var_32, 0, 0, 15727360, 0, 13628160, 0, _var_34, _var_38, _var_41, _var_42, 274877906944, 127, 2147549185, 0, _var_43, _var_45, _var_46, &_var_0, &_var_1, &_var_2, &_var_3, &_var_4, &_var_5, &_var_6, &_var_7, &_var_8, &_var_9, &_var_10, &_var_11, &_var_12, &_var_13, &_var_14, &_var_15, &_var_16, &_var_17, &_var_18, &_var_19, &_var_20, &_var_21, &_var_22, &_var_23);
      }
    }
  }
}

_ABI(SystemV_x86_64)
void unreserved___do_global_ctors_aux(generic64_t argument_0, generic64_t argument_1) {
  if (segment_0x406fd0_Generic64_2224.unreserved__ctors.offset_0 != (pointer_or_number64_t) -1) {
    generic64_t _var_0;
    generic64_t _var_1;
    _var_0 = 0;
    _var_1 = segment_0x406fd0_Generic64_2224.unreserved__ctors.offset_0;
    do {
      ((cabifunction_761 *) _var_1)();
      _var_1 = *((generic64_t *) (4222920 - (_var_0 << 3)));
      _var_0 = _var_0 + 1;
    } while (_var_1 != (pointer_or_number64_t) -1);
  }
}

_ABI(SystemV_x86_64)
struct_755 function_0x404091_Code_x86_64(void) {
  struct _PACKED struct_564 {
    uint8_t padding_at_0[8];
  } _stack;
  struct_755 _var_0;
  unreserved___do_global_dtors_aux();
  _var_0.offset_0 = *((generic64_t *) &_stack);
  return _var_0;
}

