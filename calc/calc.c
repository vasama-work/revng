_ABI(SystemV_x86_64)
generic64_t function_0x401000_Code_x86_64(generic64_t argument_0, generic64_t argument_1, union_605 *argument_2) {
  struct _PACKED struct_552 {
    uint8_t padding_at_0[8];
  };
  frame_dummy();
  unreserved___do_global_ctors_aux(argument_0, argument_1);
  return /* undef */ (generic64_t){0};
}

_ABI(SystemV_x86_64)
void frame_dummy(void) {
  return;
}

_ABI(SystemV_x86_64)
void unreserved___do_global_ctors_aux(generic64_t argument_0, generic64_t argument_1) {
  struct _PACKED struct_553 {
    uint8_t padding_at_0[24];
  };
  generic64_t _var_0;
  generic64_t _var_1;
  generic8_t _var_2[8];
  *(generic64_t *)&_var_2 = *(generic64_t *)4222928ul;
  if (*(generic64_t *)&_var_2 == 18446744073709551615ul) {
  } else {
    _var_0 = 0ul;
    _var_1 = *(generic64_t *)&_var_2;
  _label_0:
    ((cabifunction_761 *)_var_1)();
    _var_1 = *(generic64_t *)(4222920ul - (_var_0 << 3ul));
    _var_0 = _var_0 + 1ul;
    if (_var_1 == 18446744073709551615ul) {
    } else {
      goto _label_0;
    }
  }
  return;
}

_ABI(SystemV_x86_64) _Noreturn
void function_0x401010_Code_x86_64(void) {
  struct _PACKED struct_554 {
    generic64_t offset_0;
  };
  generic8_t _var_0[8];
  generic8_t _var_1[4];
  unreserved__start_c((int64_t *)&_var_0 + 1ul);
  *(generic64_t *)&_var_0 = 4223304ul;
  *(int32_t *)&_var_1 = unreserved___libc_start_main((cabifunction_45 *)4201122ul, (int32_t)(generic32_t)((generic64_t *)&_var_0)[1ul], (int8_t **)&_var_0 + 2ul);
  _abort((void *)"A longjmp was taken");
  __builtin_unreachable();
}

_ABI(SystemV_x86_64)
void unreserved__start_c(int64_t *p) {
  struct _PACKED struct_555 {
    uint8_t padding_at_0[8];
  };
  generic8_t _var_0[4];
  *(int32_t *)&_var_0 = unreserved___libc_start_main((cabifunction_45 *)4201122ul, (int32_t)(generic32_t)*(generic64_t *)p, (int8_t **)p + 1ul);
  return;
}

_ABI(SystemV_x86_64)
int32_t unreserved___libc_start_main(cabifunction_45 *main_, int32_t argc, int8_t **argv) {
  struct _PACKED struct_556 {
    uint8_t padding_at_0[40];
  };
  generic8_t _var_0[8];
  generic8_t _var_1[8];
  generic8_t _var_2[8];
  unreserved___init_libc((int8_t **)((generic64_t)((int64_t)((generic64_t)(uint64_t)(uint32_t)argc << 32ul) >> 29l) + (generic64_t)argv) + 1ul, (int8_t *)*(generic64_t *)argv);
  *(generic64_t *)&_var_0 = function_0x401000_Code_x86_64((generic64_t)((int64_t)((generic64_t)(uint64_t)(uint32_t)argc << 32ul) >> 29l) + (generic64_t)argv + 8ul, *(generic64_t *)argv, (union union_605 *)argv);
  struct rawfunction_25 _var_3 = ((rawfunction_25 *)main_)((pointer_or_number64_t)/* undef */ (generic64_t){0}, (pointer_or_number64_t)((generic64_t)((int64_t)((generic64_t)(uint64_t)(uint32_t)argc << 32ul) >> 29l) + (generic64_t)argv + 8ul), (pointer_or_number64_t)argv, (pointer_or_number64_t)(uint64_t)(uint32_t)argc, (pointer_or_number64_t)/* undef */ (generic64_t){0}, (pointer_or_number64_t)/* undef */ (generic64_t){0});
  *(pointer_or_number64_t *)&_var_1 = _var_3.;
  *(pointer_or_number64_t *)&_var_2 = _var_3.;
  exit((int32_t)(generic32_t)*(generic64_t *)&_var_1);
  __builtin_unreachable();
}

_ABI(SystemV_x86_64)
void unreserved___init_libc(int8_t **envp, int8_t *pn) {
  struct _PACKED struct_561 {
    union _PACKED union_632 {
      struct _PACKED struct_633 {
        uint8_t padding_at_0[32];
        union _PACKED union_619 {
          struct _PACKED struct_620 {
            uint8_t padding_at_0[128];
            generic64_t offset_128;
          } member_0;
          struct _PACKED struct_621 {
            uint8_t padding_at_0[256];
            generic64_t offset_256;
          } member_1;
          struct _PACKED struct_622 {
            uint8_t padding_at_0[48];
            generic64_t offset_48;
          } member_2;
          struct _PACKED struct_623 {
            uint8_t padding_at_0[200];
            generic64_t offset_200;
          } member_3;
          struct _PACKED struct_624 {
            uint8_t padding_at_0[96];
            generic64_t offset_96;
          } member_4;
          struct _PACKED struct_625 {
            uint8_t padding_at_0[88];
            generic64_t offset_88;
          } member_5;
          struct _PACKED struct_626 {
            uint8_t padding_at_0[112];
            generic64_t offset_112;
          } member_6;
          struct _PACKED struct_627 {
            uint8_t padding_at_0[104];
            generic64_t offset_104;
          } member_7;
          struct _PACKED struct_628 {
            uint8_t padding_at_0[184];
            generic64_t offset_184;
          } member_8;
          struct _PACKED struct_629 {
            uint8_t padding_at_0[24];
            struct _PACKED struct_692 {
              generic32_t offset_0;
              uint8_t padding_at_4[12];
              generic64_t offset_16;
              uint8_t padding_at_24[8];
              generic64_t offset_32;
              generic64_t offset_40;
              generic64_t offset_48;
            } *offset_24;
          } member_9;
          struct _PACKED struct_630 {
            uint8_t padding_at_0[40];
            generic64_t offset_40;
          } member_10;
          struct _PACKED struct_631 {
            uint8_t padding_at_0[32];
            generic64_t offset_32;
          } member_11;
          generic32_t member_12[76];
        } offset_32;
      } member_0;
      struct _PACKED struct_636 {
        uint8_t padding_at_0[14];
        struct _PACKED struct_635 {
          struct _PACKED struct_634 {
            generic8_t offset_0;
            uint8_t padding_at_1[7];
          } offset_0[2];
          generic8_t offset_16;
        } offset_14;
      } member_1;
      struct _PACKED struct_637 {
        uint8_t padding_at_0[36];
        generic32_t offset_36[76];
      } member_2;
      struct _PACKED struct_638 {
        uint8_t padding_at_0[8];
        generic32_t offset_8[7];
      } member_3;
    } offset_0;
    uint8_t padding_at_340[4];
  };
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
  generic8_t _var_48[344];
  generic32_t _var_49;
  generic64_t _var_50;
  generic32_t _var_51;
  generic64_t _var_52;
  generic64_t _var_53;
  generic64_t _var_54;
  generic32_t _var_55;
  generic32_t _var_56;
  generic64_t _var_57;
  generic32_t _var_58;
  generic32_t _var_59;
  generic64_t _var_60;
  generic32_t _var_61;
  generic64_t _var_62;
  generic64_t _var_63;
  generic64_t _var_64;
  generic64_t _var_65;
  generic32_t _var_66;
  generic32_t _var_67;
  generic64_t _var_68;
  generic32_t _var_69;
  generic64_t _var_70;
  generic64_t _var_71;
  generic64_t _var_72;
  generic64_t _var_73;
  generic64_t _var_74;
  generic64_t _var_75;
  generic64_t _var_76;
  generic64_t _var_77;
  generic64_t _var_78;
  generic64_t _var_79;
  generic64_t _var_80;
  generic64_t _var_81;
  generic8_t _var_82[8];
  generic8_t _var_83[8];
  generic8_t _var_84[8];
  generic8_t _var_85[8];
  generic8_t _var_86[8];
  generic8_t _var_87[8];
  generic8_t _var_88[8];
  generic8_t _var_89[8];
  generic8_t _var_90[8];
  generic8_t _var_91;
  generic8_t _var_92[8];
  generic8_t _var_93[8];
  generic8_t _var_94;
  *(generic64_t *)4224456ul = (generic64_t)envp;
  _var_81 = (generic64_t)&_var_48 + 32ul;
  _var_80 = 0ul;
_label_0:
  *(generic64_t *)&_var_82 = _var_80;
  *(generic64_t *)&_var_83 = _var_81;
  *(generic32_t *)*(generic64_t *)&_var_83 = 0u;
  _var_81 = *(generic64_t *)&_var_83 + 4ul;
  _var_80 = *(generic64_t *)&_var_82 + 1ul;
  if (*(generic64_t *)&_var_82 == 75ul) {
    _var_79 = 0ul;
  _label_1:
    *(generic64_t *)&_var_84 = _var_79 << 3ul;
    *(generic64_t *)&_var_85 = *(generic64_t *)(*(generic64_t *)&_var_84 + (generic64_t)envp);
    _var_79 = _var_79 + 1ul;
    if (*(generic64_t *)&_var_85 == 0ul) {
      *(generic64_t *)4225040ul = *(generic64_t *)&_var_84 + 8ul + (generic64_t)envp;
      *(generic64_t *)&_var_86 = *(generic64_t *)(*(generic64_t *)&_var_84 + 8ul + (generic64_t)envp);
      if (*(generic64_t *)&_var_86 == 0ul) {
        *(generic64_t *)4224992ul = ((generic64_t *)&_var_48)[20ul];
        *(generic64_t *)4225112ul = ((generic64_t *)&_var_48)[36ul];
        *(generic64_t *)&_var_88 = ((generic64_t *)&_var_48)[10ul];
        *(generic64_t *)4225056ul = *(generic64_t *)&_var_88;
        if ((generic64_t)pn == 0ul) {
          unreserved___init_tls((size_t *)&_var_48 + 4ul);
          dummy1((void *)((generic64_t *)&_var_48)[29ul]);
          if (((generic64_t *)&_var_48)[15ul] == ((generic64_t *)&_var_48)[16ul]) {
            if (((generic64_t *)&_var_48)[17ul] == ((generic64_t *)&_var_48)[18ul]) {
              if (((generic64_t *)&_var_48)[27ul] == 0ul) {
              } else {
                _var_71 = (generic64_t)&_var_48 + 8ul;
                _var_70 = 0ul;
              _label_2:
                *(generic64_t *)&_var_92 = _var_70;
                *(generic64_t *)&_var_93 = _var_71;
                *(generic32_t *)*(generic64_t *)&_var_93 = 0u;
                _var_71 = *(generic64_t *)&_var_93 + 4ul;
                _var_70 = *(generic64_t *)&_var_92 + 1ul;
                if (*(generic64_t *)&_var_92 == 5ul) {
                  ((generic32_t *)&_var_48)[4ul] = 1u;
                  ((generic32_t *)&_var_48)[6ul] = 2u;
                  helper_syscall_wrapper((void *)0ul, 2u, 4201492ul, /* undef */ (generic64_t){0}, (generic64_t)&_var_48 + 8ul, /* undef */ (generic64_t){0}, 7ul, /* undef */ (generic64_t){0}, 0ul, 0ul, (generic64_t)&_var_48 + 8ul, 0ul, 3ul, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42, (void *)&_var_43, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47);
                  _var_59 = _var_29;
                  _var_60 = _var_30;
                  _var_61 = _var_32;
                  _var_63 = _var_34;
                  _var_64 = _var_38;
                  _var_65 = _var_41;
                  _var_66 = _var_42;
                  _var_67 = _var_43;
                  _var_68 = _var_45;
                  _var_69 = _var_46;
                  _var_62 = 0ul;
                _label_3:
                  _var_49 = _var_59;
                  _var_50 = _var_60;
                  _var_51 = _var_61;
                  _var_52 = _var_63;
                  _var_53 = _var_64;
                  _var_54 = _var_65;
                  _var_55 = _var_66;
                  _var_56 = _var_67;
                  _var_57 = _var_68;
                  _var_58 = _var_69;
                  if ((*(generic8_t *)((generic64_t)&_var_48 + 14ul + (_var_62 << 3ul)) & 32u) == 0u) {
                  } else {
                    helper_syscall_wrapper((void *)0ul, 2u, 4201515ul, /* undef */ (generic64_t){0}, (generic64_t)&_var_48 + 8ul, /* undef */ (generic64_t){0}, 2ul, /* undef */ (generic64_t){0}, _var_62, 0ul, 4214788ul, 0ul, 2ul, _var_59, _var_60, _var_61, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, _var_63, _var_64, _var_65, _var_66, 274877906944ul, 127u, 2147549185ul, 0ul, _var_67, _var_68, _var_69, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
                    _var_49 = _var_5;
                    _var_50 = _var_6;
                    _var_51 = _var_8;
                    _var_52 = _var_10;
                    _var_53 = _var_14;
                    _var_54 = _var_17;
                    _var_55 = _var_18;
                    _var_56 = _var_19;
                    _var_57 = _var_21;
                    _var_58 = _var_22;
                  }
                  *(int8_t *)&_var_94 = _var_62 == 2ul;
                  _var_62 = _var_62 + 1ul;
                  if (_var_94) {
                    *(generic32_t *)4225032ul = 1u;
                  } else {
                    goto _label_3;
                  }
                } else {
                  goto _label_2;
                }
              }
            } else {
            }
          } else {
          }
        } else {
          *(generic64_t *)4223408ul = (generic64_t)pn;
          _var_75 = (generic64_t)pn;
          _var_76 = *(generic64_t *)&_var_88;
        _label_4:
          _var_73 = _var_75;
          _var_74 = _var_76;
          *(generic64_t *)4223416ul = _var_73;
          *(generic64_t *)&_var_89 = _var_73 + 1ul;
          _var_72 = 0ul;
        _label_5:
          *(generic64_t *)&_var_90 = *(generic64_t *)&_var_89 + _var_72;
          _var_91 = *(generic8_t *)_var_73;
          if (_var_91 == 0u) {
          } else {
            _var_73 = _var_73 + 1ul;
            _var_74 = _var_74 & 18446744073709551360ul | (generic64_t)(uint64_t)(uint8_t)_var_91;
            _var_72 = _var_72 + 1ul;
            if (_var_91 == 47u) {
              _var_75 = *(generic64_t *)&_var_90;
              _var_76 = _var_74;
              goto _label_4;
            } else {
              goto _label_5;
            }
          }
        }
      } else {
        _var_77 = 0ul;
        _var_78 = *(generic64_t *)&_var_86;
      _label_6:
        *(generic64_t *)&_var_87 = _var_77;
        if ((generic8_t)((uint64_t)_var_78 > 37ul)) {
        } else {
          ((generic64_t *)((_var_78 << 3ul) + (generic64_t)&_var_48))[4ul] = *(generic64_t *)(*(generic64_t *)&_var_84 + ((generic64_t)envp + 16ul) + (*(generic64_t *)&_var_87 << 4ul));
        }
        _var_78 = *(generic64_t *)(*(generic64_t *)&_var_84 + ((generic64_t)envp + 24ul) + (*(generic64_t *)&_var_87 << 4ul));
        _var_77 = *(generic64_t *)&_var_87 + 1ul;
        if (_var_78 == 0ul) {
        } else {
          goto _label_6;
        }
      }
    } else {
      goto _label_1;
    }
  } else {
    goto _label_0;
  }
  return;
}

_ABI(SystemV_x86_64) _Noreturn
void exit(int32_t code) {
  struct _PACKED struct_563 {
    uint8_t padding_at_0[24];
  };
  dummy_();
  function_0x404091_Code_x86_64();
  unreserved___stdio_exit();
  unreserved__Exit(code);
  __builtin_unreachable();
}

_ABI(SystemV_x86_64)
void deregister_tm_clones(void) {
  return;
}

_ABI(SystemV_x86_64)
void register_tm_clones(void) {
  return;
}

_ABI(SystemV_x86_64)
void unreserved___do_global_dtors_aux(void) {
  struct _PACKED struct_557 {
    uint8_t padding_at_0[8];
    generic64_t offset_8;
    uint8_t padding_at_16[8];
  };
  generic8_t _var_0[24];
  if (*(generic8_t *)4223328ul == 0u) {
    ((generic64_t *)&_var_0)[1ul] = 4222944ul;
    deregister_tm_clones();
    *(generic8_t *)4223328ul = 1u;
  } else {
  }
  return;
}

_ABI(SystemV_x86_64) _Noreturn
void function_0x401171_Code_x86_64(void) {
  _abort((void *)"A longjmp was taken");
  __builtin_unreachable();
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
  };
  generic8_t _var_0[248];
  generic32_t _var_1;
  generic32_t _var_2;
  generic32_t _var_3;
  generic16_t _var_4;
  generic32_t _var_5;
  generic64_t _var_6;
  generic8_t _var_7;
  generic8_t _var_8;
  generic8_t _var_9;
  generic8_t _var_10[4];
  ((generic64_t *)&_var_0)[1ul] = (generic64_t)buffer;
  *(generic64_t *)&_var_0 = (generic64_t)size;
  ((generic32_t *)&_var_0)[59ul] = 4294967295u;
  ((generic32_t *)&_var_0)[58ul] = 0u;
  ((generic32_t *)&_var_0)[57ul] = 0u;
  ((generic32_t *)&_var_0)[56ul] = 0u;
  if ((generic8_t)((uint64_t)*(generic64_t *)&_var_0 > (uint64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[58ul])) {
    _var_6 = (generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[58ul];
  _label_0:
    switch (*(generic8_t *)(((generic64_t *)&_var_0)[1ul] + _var_6)) {
      case 40u: {
        if (((generic32_t *)&_var_0)[56ul] == 0u) {
          ((generic32_t *)&_var_0)[57ul] = 0u;
          ((generic32_t *)&_var_0)[59ul] = ((generic32_t *)&_var_0)[59ul] + 1u;
          if (((generic32_t *)&_var_0)[59ul] == 10u) {
            _var_1 = 666u;
          } else {
            ((generic32_t *)&_var_0)[55ul] = 0u;
            _var_5 = 0u;
          _label_1:
            *(generic8_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 16ul) + (generic64_t)(uint64_t)(uint32_t)_var_5) = 0u;
            _var_5 = ((generic32_t *)&_var_0)[55ul] + 1u;
            ((generic32_t *)&_var_0)[55ul] = _var_5;
            if ((generic8_t)((uint32_t)_var_5 > 19u)) {
              ((generic32_t *)&_var_0)[58ul] = ((generic32_t *)&_var_0)[58ul] + 1u;
              if (*(generic64_t *)&_var_0 == (generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[58ul]) {
              } else {
                _var_8 = *(generic8_t *)(((generic64_t *)&_var_0)[1ul] + (generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[58ul]);
                _var_4 = 1u;
                switch (_var_8) {
                  case 43u: {
                    _var_4 = 2u;
                    ((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775697ul] = _var_4;
                    ((generic8_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[18446744073709551392ul] = *(generic8_t *)(((generic64_t *)&_var_0)[1ul] + (generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[58ul]);
                    ((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775698ul] = 0u;
                    ((generic32_t *)&_var_0)[58ul] = ((generic32_t *)&_var_0)[58ul] + 1u;
                    if (*(generic64_t *)&_var_0 == (generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[58ul]) {
                    } else if (*(generic8_t *)(((generic64_t *)&_var_0)[1ul] + (generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[58ul]) == 32u) {
                      ((generic32_t *)&_var_0)[58ul] = ((generic32_t *)&_var_0)[58ul] + 1u;
                      _var_6 = (generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[58ul];
                      if ((generic8_t)((uint64_t)*(generic64_t *)&_var_0 > (uint64_t)_var_6)) {
                        goto _label_0;
                      } else {
                        _var_1 = ((generic32_t *)&_var_0)[57ul];
                      }
                    } else {
                    }
                  }
                  case 45u: {
                  }
                  case 42u: {
                  }
                  case 38u: {
                  }
                  case 124u: {
                  }
                  case 94u: {
                  }
                  case 126u: {
                  }
                  case 33u: {
                  }
                  case 63u:
                    _var_4 = 3u;
                  default: {
                  }
                }
              }
            } else {
              goto _label_1;
            }
          }
        } else {
        }
      }
      case 45u: {
        *(int8_t *)&_var_7 = ((generic32_t *)&_var_0)[56ul] == 0u;
        _var_3 = 2u;
        if (_var_7) {
          ((generic32_t *)&_var_0)[56ul] = _var_3;
        } else {
        }
      }
      default: {
        if ((generic8_t)((uint8_t)(*(generic8_t *)(((generic64_t *)&_var_0)[1ul] + _var_6) + 198u) < 246u)) {
          switch (*(generic8_t *)(((generic64_t *)&_var_0)[1ul] + _var_6)) {
            case 32u: {
              if (((generic32_t *)&_var_0)[59ul] == 4294967295u) {
              } else {
                ((generic32_t *)(((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 5ul + (generic64_t)(uint64_t)(uint16_t)((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775698ul] << 2ul) + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] = ((generic32_t *)&_var_0)[57ul];
                ((generic32_t *)&_var_0)[56ul] = 0u;
                ((generic32_t *)&_var_0)[57ul] = 0u;
                ((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775698ul] = ((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775698ul] + 1u;
                if (((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775698ul] == ((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775697ul]) {
                } else {
                }
              }
            }
            case 41u: {
              if (((generic32_t *)&_var_0)[59ul] == 4294967295u) {
              } else {
                ((generic32_t *)(((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 5ul + (generic64_t)(uint64_t)(uint16_t)((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775698ul] << 2ul) + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] = ((generic32_t *)&_var_0)[57ul];
                ((generic32_t *)&_var_0)[56ul] = 0u;
                ((generic32_t *)&_var_0)[57ul] = 0u;
                if ((generic32_t)(uint32_t)(uint16_t)((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775698ul] + 1u == (generic32_t)(uint32_t)(uint16_t)((generic16_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[9223372036854775697ul]) {
                  switch (((generic8_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[18446744073709551392ul]) {
                    case 43u: {
                      _var_2 = ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] + ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387851ul];
                      ((generic32_t *)&_var_0)[57ul] = _var_2;
                      *(generic32_t *)&_var_10 = ((generic32_t *)&_var_0)[59ul];
                      ((generic32_t *)&_var_0)[59ul] = *(generic32_t *)&_var_10 + 4294967295u;
                      if ((generic8_t)((uint32_t)*(generic32_t *)&_var_10 > 2147483648u)) {
                      } else {
                      }
                    }
                    case 45u:
                      _var_2 = ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] - ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387851ul];
                    case 42u:
                      _var_2 = ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387851ul] * ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul];
                    case 38u:
                      _var_2 = ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] & ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387851ul];
                    case 124u:
                      _var_2 = ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] | ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387851ul];
                    case 94u:
                      _var_2 = ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] ^ ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387851ul];
                    case 63u:
                      _var_2 = *(generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul) + (((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] == 0u ? 18446744073709551408ul : 18446744073709551404ul));
                    case 126u:
                      _var_2 = ((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] ^ 4294967295u;
                    case 33u:
                      _var_2 = (generic32_t)(uint32_t)(uint8_t)(((generic32_t *)((generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[59ul] * 20ul + ((generic64_t)&_var_0 + 240ul)))[4611686018427387850ul] == 0u);
                    default: {
                    }
                  }
                } else {
                }
              }
            }
            default: {
            }
          }
        } else {
          ((generic32_t *)&_var_0)[57ul] = (generic32_t)(int32_t)(int8_t)*(generic8_t *)(((generic64_t *)&_var_0)[1ul] + _var_6) + 4294967248u + ((generic32_t *)&_var_0)[57ul] * 10u;
          *(int8_t *)&_var_9 = ((generic32_t *)&_var_0)[56ul] == 2u;
          _var_3 = 1u;
          if (_var_9) {
            ((generic32_t *)&_var_0)[57ul] = 0u - ((generic32_t *)&_var_0)[57ul];
            _var_3 = 1u;
          } else {
          }
        }
      }
    }
  } else {
  }
  return (int32_t)_var_1;
}

_ABI(SystemV_x86_64)
int32_t main(int32_t argc, int8_t **argv) {
  struct _PACKED struct_559 {
    struct _PACKED struct_686 {
      uint8_t padding_at_0[8];
      struct _PACKED struct_691 {
        union _PACKED union_690 {
          struct_688 member_0;
          generic64_t member_1;
        } offset_0;
        struct _PACKED struct_689 {
          struct_688 offset_0;
          uint8_t padding_at_2[6];
        } offset_8;
      } *offset_8;
    } *offset_0;
    uint8_t padding_at_8[4];
    generic32_t offset_12;
    uint8_t padding_at_16[8];
  };
  generic8_t _var_0[24];
  generic8_t _var_1[8];
  generic8_t _var_2[4];
  generic8_t _var_3[4];
  ((generic32_t *)&_var_0)[3ul] = (generic32_t)argc;
  *(generic64_t *)&_var_0 = (generic64_t)argv;
  *(size_t *)&_var_1 = strlen((const int8_t *)((generic64_t *)argv)[1ul]);
  *(int32_t *)&_var_2 = root((int8_t *)((generic64_t *)*(generic64_t *)&_var_0)[1ul], (size_t)*(generic64_t *)&_var_1);
  *(int32_t *)&_var_3 = printf((typedef_66)4214784ul);
  return 0;
}

_ABI(SystemV_x86_64)
size_t strlen(const int8_t *s) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic64_t _var_6;
  generic8_t _var_7[8];
  generic8_t _var_8[8];
  generic8_t _var_9;
  generic8_t _var_10[8];
  generic8_t _var_11;
  _var_4 = (generic64_t)s;
  if (((generic64_t)s & 7ul) == 0ul) {
    _var_2 = _var_4;
    if ((*(generic64_t *)_var_2 + 18374403900871474943ul & (*(generic64_t *)_var_2 ^ 18446744073709551615ul) & 9259542123273814144ul) == 0ul) {
      _var_3 = 0ul;
    _label_0:
      *(generic64_t *)&_var_8 = _var_4 + 8ul + (_var_3 << 3ul);
      *(int8_t *)&_var_9 = (*(generic64_t *)*(generic64_t *)&_var_8 + 18374403900871474943ul & (*(generic64_t *)*(generic64_t *)&_var_8 ^ 18446744073709551615ul) & 9259542123273814144ul) == 0ul;
      _var_3 = _var_3 + 1ul;
      if (_var_9) {
        goto _label_0;
      } else {
        _var_2 = *(generic64_t *)&_var_8;
        _var_1 = _var_2;
      _label_1:
        *(generic64_t *)&_var_10 = _var_1;
        *(int8_t *)&_var_11 = *(generic8_t *)*(generic64_t *)&_var_10 == 0u;
        _var_1 = *(generic64_t *)&_var_10 + 1ul;
        if (_var_11) {
          _var_0 = *(generic64_t *)&_var_10;
        } else {
          goto _label_1;
        }
      }
    } else {
    }
  } else {
    _var_5 = 0ul;
    _var_6 = (generic64_t)s;
  _label_2:
    *(generic64_t *)&_var_7 = (generic64_t)s + 1ul + _var_5;
    if (*(generic8_t *)_var_6 == 0u) {
      _var_0 = _var_6;
    } else {
      _var_6 = _var_6 + 1ul;
      _var_5 = _var_5 + 1ul;
      if ((*(generic64_t *)&_var_7 & 7ul) == 0ul) {
        _var_4 = *(generic64_t *)&_var_7;
      } else {
        goto _label_2;
      }
    }
  }
  return (size_t)(_var_0 - (generic64_t)s);
}

_ABI(SystemV_x86_64)
int32_t printf(typedef_66 fmt) {
  struct _PACKED struct_560 {
    uint8_t padding_at_0[8];
    struct _PACKED struct_616 {
      union _PACKED union_614 {
        struct _PACKED struct_615 {
          uint8_t padding_at_0[4];
          generic32_t offset_4;
        } member_0;
        generic32_t member_1;
        generic64_t member_2;
      } offset_0;
      generic64_t offset_8;
      generic64_t offset_16;
    } offset_8;
    uint8_t padding_at_32[8];
    union_605 *offset_40;
    uint8_t padding_at_48[168];
  };
  generic8_t _var_0[216];
  generic8_t _var_1[4];
  generic8_t _var_2[8];
  ((generic64_t *)&_var_0)[5ul] = (generic64_t)fmt;
  *(generic64_t *)&_var_2 = *(generic64_t *)4214800ul;
  ((generic64_t *)&_var_0)[2ul] = (generic64_t)&_var_0 + 224ul;
  ((generic32_t *)&_var_0)[2ul] = 8u;
  ((generic32_t *)&_var_0)[3ul] = 48u;
  ((generic64_t *)&_var_0)[3ul] = (generic64_t)&_var_0 + 32ul;
  *(int32_t *)&_var_1 = vfprintf((typedef_88)*(generic64_t *)&_var_2, (typedef_104)fmt, (struct unreserved___va_list_tag *)((generic64_t)&_var_0 + 8ul));
  return (int32_t)*(generic32_t *)&_var_1;
}

_ABI(SystemV_x86_64)
int32_t vfprintf(typedef_88 f, typedef_104 fmt, unreserved___va_list_tag *ap) {
  struct _PACKED struct_566 {
    uint8_t padding_at_0[12];
    generic32_t offset_12;
    union _PACKED union_640 {
      struct_639 member_0;
      struct _PACKED struct_641 {
        uint8_t padding_at_0[8];
        generic64_t offset_8;
      } member_1;
      struct _PACKED struct_642 {
        uint8_t padding_at_0[16];
        generic64_t offset_16;
      } member_2;
      generic64_t member_3;
    } offset_16;
    uint8_t padding_at_40[336];
  };
  generic8_t _var_0[376];
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic64_t _var_6;
  generic8_t _var_7[4];
  generic8_t _var_8[4];
  generic8_t _var_9[4];
  generic8_t _var_10[8];
  generic8_t _var_11[8];
  generic8_t _var_12[8];
  generic8_t _var_13[8];
  generic8_t _var_14[8];
  generic8_t _var_15;
  generic8_t _var_16[8];
  generic8_t _var_17[4];
  generic8_t _var_18;
  generic8_t _var_19;
  generic8_t _var_20[8];
  generic8_t _var_21[4];
  generic8_t _var_22[4];
  _var_6 = (generic64_t)&_var_0 + 40ul;
  _var_5 = 0ul;
_label_0:
  *(generic64_t *)&_var_12 = _var_5;
  *(generic64_t *)&_var_13 = _var_6;
  *(generic32_t *)*(generic64_t *)&_var_13 = 0u;
  _var_6 = *(generic64_t *)&_var_13 + 4ul;
  _var_5 = *(generic64_t *)&_var_12 + 1ul;
  if (*(generic64_t *)&_var_12 == 9ul) {
    *(generic64_t *)&_var_14 = ((generic64_t *)ap)[1ul];
    ((generic64_t *)&_var_0)[2ul] = *(generic64_t *)ap;
    ((generic64_t *)&_var_0)[3ul] = *(generic64_t *)&_var_14;
    ((generic64_t *)&_var_0)[4ul] = ((generic64_t *)ap)[2ul];
    *(int32_t *)&_var_7 = printf_core((FILE_ *)0ul, (const int8_t *)fmt, (va_list *)((generic64_t)&_var_0 + 16ul), (union arg *)&_var_0 + 10ul, (int32_t *)&_var_0 + 10ul);
    _var_1 = 4294967295ul;
    if ((generic8_t)((int32_t)*(generic32_t *)&_var_7 > 4294967295)) {
      _var_15 = (generic8_t)((int32_t)((generic32_t *)f)[35ul] > 4294967295);
      _var_4 = 0ul;
      if (_var_15) {
        *(int32_t *)&_var_8 = unreserved___lockfile((FILE_ *)f);
        _var_4 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_8;
      } else {
      }
      *(generic64_t *)&_var_16 = _var_4;
      *(generic32_t *)&_var_17 = *(generic32_t *)f;
      _var_18 = ((generic8_t *)f)[138ul];
      ((generic32_t *)&_var_0)[3ul] = *(generic32_t *)&_var_17 & 32u;
      if ((generic8_t)((int8_t)_var_18 > 0)) {
      } else {
        *(generic32_t *)f = *(generic32_t *)&_var_17 & 4294967263u;
      }
      *(int8_t *)&_var_19 = ((generic64_t *)f)[12ul] == 0ul;
      _var_3 = 0ul;
      if (_var_19) {
        _var_3 = ((generic64_t *)f)[11ul];
        ((generic64_t *)f)[12ul] = 80ul;
        ((generic64_t *)f)[11ul] = (generic64_t)&_var_0 + 80ul;
        ((generic64_t *)f)[7ul] = (generic64_t)&_var_0 + 80ul;
        ((generic64_t *)f)[5ul] = (generic64_t)&_var_0 + 80ul;
        ((generic64_t *)f)[4ul] = (generic64_t)&_var_0 + 160ul;
      } else {
      }
      *(int32_t *)&_var_9 = printf_core((FILE_ *)f, (const int8_t *)fmt, (va_list *)((generic64_t)&_var_0 + 16ul), (union arg *)&_var_0 + 10ul, (int32_t *)&_var_0 + 10ul);
      _var_2 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_9;
      if (_var_3 == 0ul) {
      } else {
        struct rawfunction_25 _var_23 = ((rawfunction_25 *)((generic64_t *)f)[9ul])((pointer_or_number64_t)((generic64_t)&_var_0 + 160ul), 0ul, 0ul, (pointer_or_number64_t)f, (pointer_or_number64_t)((generic64_t)&_var_0 + 40ul), (pointer_or_number64_t)/* undef */ (generic64_t){0});
        *(pointer_or_number64_t *)&_var_10 = _var_23.;
        *(pointer_or_number64_t *)&_var_11 = _var_23.;
        *(generic64_t *)&_var_20 = ((generic64_t *)f)[5ul];
        ((generic64_t *)f)[11ul] = _var_3;
        ((generic64_t *)f)[12ul] = 0ul;
        _var_2 = *(generic64_t *)&_var_20 == 0ul ? 4294967295ul : (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_9;
        ((generic64_t *)f)[4ul] = 0ul;
        ((generic64_t *)f)[7ul] = 0ul;
        ((generic64_t *)f)[5ul] = 0ul;
      }
      *(generic32_t *)&_var_21 = *(generic32_t *)f;
      *(generic32_t *)&_var_22 = ((generic32_t *)&_var_0)[3ul];
      _var_1 = (*(generic32_t *)&_var_21 & 32u) == 0u ? _var_2 : 4294967295ul;
      *(generic32_t *)f = *(generic32_t *)&_var_22 | *(generic32_t *)&_var_21;
      if (*(generic64_t *)&_var_16 == 0ul) {
      } else {
        unreserved___unlockfile((FILE_ *)f);
      }
    } else {
    }
  } else {
    goto _label_0;
  }
  return (int32_t)(generic32_t)_var_1;
}

_ABI(SystemV_x86_64)
void dummy(void) {
  return;
}

_ABI(SystemV_x86_64)
void dummy1(void *p) {
  return;
}

_ABI(SystemV_x86_64)
void unreserved___init_tls(size_t *aux) {
  struct _PACKED struct_562 {
    uint8_t padding_at_0[1];
  };
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
  generic32_t _var_48;
  generic64_t _var_49;
  generic32_t _var_50;
  generic64_t _var_51;
  generic64_t _var_52;
  generic64_t _var_53;
  generic64_t _var_54;
  generic64_t _var_55;
  generic32_t _var_56;
  generic64_t _var_57;
  generic64_t _var_58;
  generic32_t _var_59;
  generic64_t _var_60;
  generic32_t _var_61;
  generic64_t _var_62;
  generic64_t _var_63;
  generic64_t _var_64;
  generic64_t _var_65;
  generic64_t _var_66;
  generic64_t _var_67;
  generic64_t _var_68;
  generic64_t _var_69;
  generic64_t _var_70;
  generic64_t _var_71;
  generic8_t _var_72[8];
  generic8_t _var_73[8];
  generic8_t _var_74[8];
  generic8_t _var_75[8];
  generic8_t _var_76[8];
  generic8_t _var_77;
  generic8_t _var_78[8];
  generic8_t _var_79[8];
  generic8_t _var_80[8];
  generic8_t _var_81[8];
  generic8_t _var_82[8];
  generic8_t _var_83[8];
  generic8_t _var_84[8];
  generic8_t _var_85[8];
  generic8_t _var_86[4];
  generic8_t _var_87[8];
  generic8_t _var_88[4];
  generic8_t _var_89[8];
  generic8_t _var_90[8];
  generic8_t _var_91[8];
  generic8_t _var_92[8];
  generic8_t _var_93[4];
  generic8_t _var_94[8];
  generic8_t _var_95[8];
  generic8_t _var_96[4];
  generic8_t _var_97[8];
  generic8_t _var_98[4];
  *(generic64_t *)&_var_75 = ((generic64_t *)aux)[3ul];
  *(generic64_t *)&_var_76 = ((generic64_t *)aux)[5ul];
  _var_63 = 0ul;
  _var_64 = 0ul;
  if (*(generic64_t *)&_var_76 == 0ul) {
    *(generic64_t *)&_var_79 = _var_64;
    *(generic64_t *)&_var_80 = _var_65;
    if (*(generic64_t *)&_var_79 == 0ul) {
      _var_62 = *(generic64_t *)4225144ul;
    } else {
      *(generic64_t *)&_var_81 = ((generic64_t *)*(generic64_t *)&_var_79)[4ul];
      *(generic64_t *)4225120ul = _var_63 + ((generic64_t *)*(generic64_t *)&_var_79)[2ul];
      *(generic64_t *)4225128ul = *(generic64_t *)&_var_81;
      *(generic64_t *)&_var_82 = ((generic64_t *)*(generic64_t *)&_var_79)[5ul];
      _var_62 = ((generic64_t *)*(generic64_t *)&_var_79)[6ul];
      *(generic64_t *)&_var_83 = _var_62;
      *(generic64_t *)4225136ul = *(generic64_t *)&_var_82;
      *(generic64_t *)4225144ul = *(generic64_t *)&_var_83;
    }
    *(generic64_t *)&_var_84 = _var_62;
    *(generic64_t *)&_var_85 = *(generic64_t *)4225136ul;
    *(generic64_t *)4225136ul = (*(generic64_t *)&_var_84 + 18446744073709551615ul & 0ul - (*(generic64_t *)4225120ul + *(generic64_t *)&_var_85)) + *(generic64_t *)&_var_85;
    if ((generic8_t)((uint64_t)*(generic64_t *)&_var_84 > 7ul)) {
    } else {
      *(generic64_t *)4225144ul = 8ul;
    }
    _var_51 = *(generic64_t *)4225144ul;
    *(generic64_t *)4225048ul = _var_51 + *(generic64_t *)4225136ul + 359ul & 18446744073709551608ul;
    _var_48 = 4294967295u;
    _var_49 = 514ul;
    _var_50 = 4243635u;
    _var_52 = 4224480ul;
    _var_53 = 0ul;
    _var_54 = 0ul;
    _var_55 = 0ul;
    _var_56 = 65535u;
    _var_57 = *(generic64_t *)&_var_80;
    _var_58 = *(generic64_t *)&_var_75;
    _var_59 = 0u;
    _var_60 = 0ul;
    _var_61 = 4294967295u;
    if ((generic8_t)((uint64_t)*(generic64_t *)4225048ul > 472ul)) {
      helper_syscall_wrapper((void *)0ul, 2u, 4208633ul, 34ul, 18446744073709551615ul, 0ul, 9ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, *(generic64_t *)&_var_85, 0ul, 3ul, *(generic64_t *)4225048ul, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42, (void *)&_var_43, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47);
      _var_52 = _var_27;
      _var_48 = _var_29;
      _var_49 = _var_30;
      _var_50 = _var_32;
      _var_53 = _var_34;
      _var_54 = _var_38;
      _var_55 = _var_41;
      _var_56 = _var_42;
      _var_59 = _var_43;
      _var_60 = _var_45;
      _var_61 = _var_46;
      _var_51 = 3ul;
      _var_57 = 0ul;
      _var_58 = 18446744073709551615ul;
    } else {
    }
    *(generic32_t *)&_var_86 = _var_48;
    *(generic64_t *)&_var_87 = _var_49;
    *(generic32_t *)&_var_88 = _var_50;
    *(generic64_t *)&_var_89 = _var_51;
    *(generic64_t *)&_var_90 = _var_53;
    *(generic64_t *)&_var_91 = _var_54;
    *(generic64_t *)&_var_92 = _var_55;
    *(generic32_t *)&_var_93 = _var_56;
    *(generic64_t *)&_var_94 = _var_57;
    *(generic64_t *)&_var_95 = _var_58;
    *(generic32_t *)&_var_96 = _var_59;
    *(generic64_t *)&_var_97 = _var_60;
    *(generic32_t *)&_var_98 = _var_61;
    *(void **)&_var_72 = unreserved___copy_tls((uint8_t *)_var_52);
    *(generic64_t *)*(generic64_t *)&_var_72 = *(generic64_t *)&_var_72;
    *(generic64_t *)&_var_73 = unreserved___set_thread_area((struct struct_703 *)*(generic64_t *)&_var_72);
    *(generic64_t *)&_var_74 = lshift(*(generic64_t *)&_var_73 & 4294967295ul, 4294967272u);
    if ((generic32_t)*(generic64_t *)&_var_73 == 0u) {
      *(generic32_t *)4225024ul = 1u;
    } else {
    }
  } else {
    _var_68 = 0ul;
    _var_69 = 0ul;
    _var_70 = 0ul;
    _var_71 = *(generic64_t *)&_var_75;
  _label_0:
    *(generic64_t *)&_var_78 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)_var_71;
    if (*(generic32_t *)_var_71 == 6u) {
      _var_67 = *(generic64_t *)&_var_75 - ((generic64_t *)_var_71)[2ul];
      _var_66 = _var_69;
    } else {
      _var_66 = *(generic32_t *)_var_71 == 7u ? _var_71 : _var_69;
      _var_67 = _var_70;
    }
    _var_71 = _var_71 + ((generic64_t *)aux)[4ul];
    *(int8_t *)&_var_77 = *(generic64_t *)&_var_76 == _var_68 + 1ul;
    _var_68 = _var_68 + 1ul;
    if (_var_77) {
      _var_63 = _var_67;
      _var_64 = _var_66;
      _var_65 = *(generic64_t *)&_var_78;
    } else {
      goto _label_0;
    }
  }
  helper_syscall_wrapper((void *)0ul, 2u, 4208685ul, 34ul, *(generic64_t *)&_var_95, *(generic64_t *)&_var_94, 218ul, /* undef */ (generic64_t){0}, *(generic64_t *)&_var_72, *(generic64_t *)&_var_85, *(generic64_t *)&_var_72 + 56ul, *(generic64_t *)&_var_89, *(generic64_t *)4225048ul, *(generic32_t *)&_var_86, *(generic64_t *)&_var_87, *(generic32_t *)&_var_88, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, *(generic64_t *)&_var_90, *(generic64_t *)&_var_91, *(generic64_t *)&_var_92, *(generic32_t *)&_var_93, 274877906944ul, 127u, 2147549185ul, 0ul, *(generic32_t *)&_var_96, *(generic64_t *)&_var_97, *(generic32_t *)&_var_98, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
  ((generic32_t *)*(generic64_t *)&_var_72)[14ul] = (generic32_t)_var_3;
  ((generic64_t *)*(generic64_t *)&_var_72)[32ul] = 4225064ul;
  ((generic64_t *)*(generic64_t *)&_var_72)[28ul] = *(generic64_t *)&_var_72 + 224ul;
  return;
}

_ABI(SystemV_x86_64)
void *unreserved___copy_tls(uint8_t *mem) {
  struct _PACKED struct_573 {
    uint8_t padding_at_0[8];
  };
  generic64_t _var_0;
  generic8_t _var_1[8];
  generic8_t _var_2;
  generic8_t _var_3[8];
  generic8_t _var_4[8];
  generic8_t _var_5[8];
  generic8_t _var_6[8];
  generic8_t _var_7[8];
  generic8_t _var_8[8];
  *(int8_t *)&_var_2 = *(generic64_t *)4225120ul == 0ul;
  _var_0 = (generic64_t)mem;
  if (_var_2) {
  } else {
    *(generic64_t *)&_var_3 = *(generic64_t *)4225048ul;
    *(generic64_t *)&_var_4 = *(generic64_t *)4225144ul;
    *(generic64_t *)mem = 1ul;
    *(generic64_t *)&_var_5 = *(generic64_t *)4225120ul;
    _var_0 = *(generic64_t *)&_var_3 + (generic64_t)mem + 18446744073709551280ul & 0ul - *(generic64_t *)&_var_4;
    *(generic64_t *)&_var_6 = _var_0;
    *(generic64_t *)&_var_7 = *(generic64_t *)4225128ul;
    *(generic64_t *)&_var_8 = *(generic64_t *)&_var_6 - *(generic64_t *)4225136ul;
    ((generic64_t *)*(generic64_t *)&_var_6)[1ul] = (generic64_t)mem;
    ((generic64_t *)*(generic64_t *)&_var_6)[41ul] = (generic64_t)mem;
    ((generic64_t *)mem)[1ul] = *(generic64_t *)&_var_8;
    *(struct struct_718 **)&_var_1 = memcpy((struct struct_718 *)*(generic64_t *)&_var_8, (union union_596 *)*(generic64_t *)&_var_5, *(generic64_t *)&_var_7);
  }
  return (void *)_var_0;
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
  helper_syscall_wrapper((void *)0ul, 2u, 4209742ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 158ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 4098ul, /* undef */ (generic64_t){0}, (generic64_t)argument_0, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
  return _var_3;
}

_ABI(SystemV_x86_64)
void dummy_(void) {
  return;
}

_ABI(SystemV_x86_64)
struct_755 function_0x404091_Code_x86_64(void) {
  struct _PACKED struct_564 {
    uint8_t padding_at_0[8];
  };
  unreserved___do_global_dtors_aux();
  generic8_t _var_0[16];
  ((generic8_t *)&_var_0)[1ul] = 0u;
  ((generic8_t *)&_var_0)[2ul] = 0u;
  ((generic8_t *)&_var_0)[3ul] = 0u;
  ((generic8_t *)&_var_0)[4ul] = 0u;
  ((generic8_t *)&_var_0)[5ul] = 0u;
  ((generic8_t *)&_var_0)[6ul] = 0u;
  ((generic8_t *)&_var_0)[7ul] = 0u;
  return *(struct struct_755 *)_var_0;
}

_ABI(SystemV_x86_64)
void unreserved___stdio_exit(void) {
  struct _PACKED struct_565 {
    uint8_t padding_at_0[8];
  };
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
  generic8_t _var_25[8];
  generic8_t _var_26[4];
  generic8_t _var_27[8];
  generic8_t _var_28[8];
  generic8_t _var_29[8];
  generic8_t _var_30[8];
  *(FILE_ ***)&_var_25 = unreserved___ofl_lock();
  if (*(generic64_t *)*(generic64_t *)&_var_25 == 0ul) {
    close_file((FILE_ *)*(generic64_t *)4224952ul);
    if (*(generic64_t *)4223040ul == 0ul) {
    } else {
      if ((generic8_t)((int32_t)((generic32_t *)*(generic64_t *)4223040ul)[35ul] > 4294967295)) {
        *(int32_t *)&_var_26 = unreserved___lockfile((FILE_ *)*(generic64_t *)4223040ul);
      } else {
      }
      if ((generic8_t)((uint64_t)((generic64_t *)*(generic64_t *)4223040ul)[5ul] > (uint64_t)((generic64_t *)*(generic64_t *)4223040ul)[7ul])) {
        struct rawfunction_25 _var_31 = ((rawfunction_25 *)((generic64_t *)*(generic64_t *)4223040ul)[9ul])((pointer_or_number64_t)/* undef */ (generic64_t){0}, 0ul, 0ul, (pointer_or_number64_t)*(generic64_t *)4223040ul, (pointer_or_number64_t)/* undef */ (generic64_t){0}, (pointer_or_number64_t)/* undef */ (generic64_t){0});
        *(pointer_or_number64_t *)&_var_27 = _var_31.;
        *(pointer_or_number64_t *)&_var_28 = _var_31.;
      } else {
      }
      if ((generic8_t)((uint64_t)((generic64_t *)*(generic64_t *)4223040ul)[1ul] < (uint64_t)((generic64_t *)*(generic64_t *)4223040ul)[2ul])) {
        helper_syscall_wrapper((void *)0ul, 2u, 4209291ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 8ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)(int32_t)((generic32_t *)*(generic64_t *)4223040ul)[30ul], 1ul, ((generic64_t *)*(generic64_t *)4223040ul)[1ul] - ((generic64_t *)*(generic64_t *)4223040ul)[2ul], 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
        if ((generic8_t)((uint64_t)_var_3 > 18446744073709547520ul)) {
          *(void **)&_var_30 = (void *)(_var_4 + 18446744073709551600ul);
          *(generic64_t *)*(void **)&_var_30 = _var_3;
          *(int32_t **)&_var_29 = unreserved___errno_location();
          *(generic32_t *)*(generic64_t *)&_var_29 = 0u - (generic32_t)*(generic64_t *)*(void **)&_var_30;
        } else {
        }
      } else {
      }
    }
  } else {
    _var_24 = *(generic64_t *)*(generic64_t *)&_var_25;
  _label_0:
    close_file((FILE_ *)_var_24);
    _var_24 = ((generic64_t *)_var_24)[14ul];
    if (_var_24 == 0ul) {
    } else {
      goto _label_0;
    }
  }
  return;
}

_ABI(SystemV_x86_64) _Noreturn
void unreserved__Exit(int32_t ec) {
  if (1u) {
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
    void *_var_48;
    void *_var_49;
    void *_var_50;
    void *_var_51;
    void *_var_52;
    void *_var_53;
    void *_var_54;
    void *_var_55;
    void *_var_56;
    void *_var_57;
    *(generic32_t **)&_var_57 = &_var_46;
    *(generic64_t **)&_var_56 = &_var_45;
    *(generic32_t **)&_var_55 = &_var_43;
    *(generic32_t **)&_var_54 = &_var_42;
    *(generic64_t **)&_var_53 = &_var_41;
    *(generic64_t **)&_var_52 = &_var_38;
    *(generic64_t **)&_var_51 = &_var_34;
    *(generic32_t **)&_var_50 = &_var_29;
    *(generic64_t **)&_var_49 = &_var_30;
    *(generic32_t **)&_var_48 = &_var_32;
    helper_syscall_wrapper((void *)0ul, 2u, 4208810ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 231ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)ec, 60ul, /* undef */ (generic64_t){0}, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, _var_50, _var_49, (void *)&_var_31, _var_48, (void *)&_var_33, _var_51, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, _var_52, (void *)&_var_39, (void *)&_var_40, _var_53, _var_54, _var_55, (void *)&_var_44, _var_56, _var_57, (void *)&_var_47);
  _label_0:
    helper_syscall_wrapper((void *)0ul, 2u, 4208820ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 60ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)ec, 60ul, /* undef */ (generic64_t){0}, *(generic32_t *)_var_50, *(generic64_t *)_var_49, *(generic32_t *)_var_48, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, *(generic64_t *)_var_51, *(generic64_t *)_var_52, *(generic64_t *)_var_53, *(generic32_t *)_var_54, 274877906944ul, 127u, 2147549185ul, 0ul, *(generic32_t *)_var_55, *(generic64_t *)_var_56, *(generic32_t *)_var_57, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
    *(generic32_t **)&_var_48 = &_var_8;
    *(generic64_t **)&_var_49 = &_var_6;
    *(generic32_t **)&_var_50 = &_var_5;
    *(generic64_t **)&_var_51 = &_var_10;
    *(generic64_t **)&_var_52 = &_var_14;
    *(generic64_t **)&_var_53 = &_var_17;
    *(generic32_t **)&_var_54 = &_var_18;
    *(generic32_t **)&_var_55 = &_var_19;
    *(generic64_t **)&_var_56 = &_var_21;
    *(generic32_t **)&_var_57 = &_var_22;
    goto _label_0;
  } else {
  }
  __builtin_unreachable();
}

_ABI(SystemV_x86_64)
FILE_ **unreserved___ofl_lock(void) {
  struct _PACKED struct_584 {
    uint8_t padding_at_0[8];
  };
  unreserved___lock((typedef_407 *)4224960ul);
  return (FILE_ **)4224968ul;
}

_ABI(SystemV_x86_64)
void close_file(FILE_ *f) {
  struct _PACKED struct_585 {
    uint8_t padding_at_0[8];
  };
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
  generic8_t _var_24[4];
  generic8_t _var_25[8];
  generic8_t _var_26[8];
  generic8_t _var_27[8];
  generic8_t _var_28[8];
  if ((generic64_t)f == 0ul) {
  } else {
    if ((generic8_t)((int32_t)((generic32_t *)f)[35ul] > 4294967295)) {
      *(int32_t *)&_var_24 = unreserved___lockfile(f);
    } else {
    }
    if ((generic8_t)((uint64_t)((generic64_t *)f)[5ul] > (uint64_t)((generic64_t *)f)[7ul])) {
      struct rawfunction_25 _var_29 = ((rawfunction_25 *)((generic64_t *)f)[9ul])((pointer_or_number64_t)/* undef */ (generic64_t){0}, 0ul, 0ul, (pointer_or_number64_t)f, (pointer_or_number64_t)/* undef */ (generic64_t){0}, (pointer_or_number64_t)/* undef */ (generic64_t){0});
      *(pointer_or_number64_t *)&_var_25 = _var_29.;
      *(pointer_or_number64_t *)&_var_26 = _var_29.;
    } else {
    }
    if ((generic8_t)((uint64_t)((generic64_t *)f)[1ul] < (uint64_t)((generic64_t *)f)[2ul])) {
      helper_syscall_wrapper((void *)0ul, 2u, 4209291ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 8ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)(int32_t)((generic32_t *)f)[30ul], 1ul, ((generic64_t *)f)[1ul] - ((generic64_t *)f)[2ul], 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
      if ((generic8_t)((uint64_t)_var_3 > 18446744073709547520ul)) {
        *(void **)&_var_28 = (void *)(_var_4 + 18446744073709551600ul);
        *(generic64_t *)*(void **)&_var_28 = _var_3;
        *(int32_t **)&_var_27 = unreserved___errno_location();
        *(generic32_t *)*(generic64_t *)&_var_27 = 0u - (generic32_t)*(generic64_t *)*(void **)&_var_28;
      } else {
      }
    } else {
    }
  }
  return;
}

_ABI(SystemV_x86_64)
int32_t unreserved___lockfile(FILE_ *f) {
  struct _PACKED struct_572 {
    uint8_t padding_at_0[24];
  };
  generic32_t _var_0;
  generic64_t _var_1;
  generic8_t _var_2[4];
  generic8_t _var_3;
  *(generic32_t *)&_var_2 = ((generic32_t *)*(generic64_t *)0ul)[14ul];
  *(int8_t *)&_var_3 = ((generic32_t *)f)[35ul] == *(generic32_t *)&_var_2;
  _var_0 = 0u;
  if (_var_3) {
  } else {
  _label_0:
    helper_lock();
    if (((generic32_t *)f)[35ul] == 0u) {
      ((generic32_t *)f)[35ul] = *(generic32_t *)&_var_2;
      _var_1 = 0ul;
    } else {
      _var_1 = (generic64_t)(uint64_t)(uint32_t)((generic32_t *)f)[35ul];
    }
    helper_unlock();
    if (_var_1 == 0ul) {
      _var_0 = 1u;
    } else {
      unreserved___wait((typedef_315 *)f + 35ul, (typedef_315 *)f + 36ul, (int32_t)(generic32_t)_var_1, 1);
      goto _label_0;
    }
  }
  return (int32_t)_var_0;
}

_ABI(SystemV_x86_64)
int32_t *unreserved___errno_location(void) {
  return (int32_t *)*(generic64_t *)0ul + 17ul;
}

_ABI(SystemV_x86_64)
int32_t printf_core(FILE_ *f, const int8_t *fmt, va_list *ap, arg *nl_arg, int32_t *nl_type) {
  struct _PACKED struct_571 {
    uint8_t padding_at_0[4];
    generic32_t offset_4;
    generic32_t offset_8;
    uint8_t padding_at_12[4];
    union _PACKED union_666 {
      struct _PACKED struct_697 {
        generic32_t offset_0;
        generic32_t offset_4;
        union _PACKED union_696 {
          generic32_t *member_0;
          generic32_t *member_1;
          generic64_t *member_2;
          generic16_t *member_3;
          generic16_t *member_4;
          generic8_t *member_5;
          generic8_t *member_6;
          generic64_t *member_7;
        } offset_8;
        generic64_t offset_16;
      } *member_0;
      struct_639 *member_1;
      struct _PACKED struct_672 {
        generic32_t offset_0;
        uint8_t padding_at_4[4];
        union _PACKED union_671 {
          generic32_t *member_0;
          generic32_t *member_1;
        } offset_8;
        generic64_t offset_16;
      } *member_2;
    } offset_16;
    struct _PACKED struct_695 {
      generic32_t offset_0;
      generic32_t offset_4;
      generic32_t offset_8;
    } *offset_24;
    union _PACKED union_667 {
      union_605 *member_0;
      generic32_t member_1;
    } offset_32;
    union _PACKED union_668 {
      struct_643 *member_0;
      struct _PACKED struct_670 {
        uint8_t padding_at_0[16];
        struct_665 offset_16;
      } *member_1;
    } offset_40;
    generic32_t offset_48;
    generic32_t offset_52;
    union_605 *offset_56;
    uint8_t padding_at_64[16];
    struct_665 offset_80;
    uint8_t padding_at_96[104];
  };
  generic8_t _var_0[200];
  generic32_t _var_1;
  generic32_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic64_t _var_6;
  generic64_t _var_7;
  generic32_t _var_8;
  generic64_t _var_9;
  generic32_t _var_10;
  generic64_t _var_11;
  generic64_t _var_12;
  generic64_t _var_13;
  generic64_t _var_14;
  generic64_t _var_15;
  generic64_t _var_16;
  generic64_t _var_17;
  generic64_t _var_18;
  generic64_t _var_19;
  generic64_t _var_20;
  generic64_t _var_21;
  generic64_t _var_22;
  generic64_t _var_23;
  generic64_t _var_24;
  generic64_t _var_25;
  generic64_t _var_26;
  generic64_t _var_27;
  generic64_t _var_28;
  generic64_t _var_29;
  generic64_t _var_30;
  generic64_t _var_31;
  generic64_t _var_32;
  generic64_t _var_33;
  generic64_t _var_34;
  generic64_t _var_35;
  generic64_t _var_36;
  generic64_t _var_37;
  generic64_t _var_38;
  generic64_t _var_39;
  generic64_t _var_40;
  generic64_t _var_41;
  generic64_t _var_42;
  generic32_t _var_43;
  generic64_t _var_44;
  generic64_t _var_45;
  generic64_t _var_46;
  generic32_t _var_47;
  generic64_t _var_48;
  generic64_t _var_49;
  generic32_t _var_50;
  generic32_t _var_51;
  generic64_t _var_52;
  generic64_t _var_53;
  generic64_t _var_54;
  generic64_t _var_55;
  generic64_t _var_56;
  generic64_t _var_57;
  generic64_t _var_58;
  generic64_t _var_59;
  generic64_t _var_60;
  generic64_t _var_61;
  generic64_t _var_62;
  generic64_t _var_63;
  generic64_t _var_64;
  generic32_t _var_65;
  generic64_t _var_66;
  generic64_t _var_67;
  generic64_t _var_68;
  generic64_t _var_69;
  generic64_t _var_70;
  generic64_t _var_71;
  generic64_t _var_72;
  generic64_t _var_73;
  generic64_t _var_74;
  generic64_t _var_75;
  generic64_t _var_76;
  generic64_t _var_77;
  generic64_t _var_78;
  generic64_t _var_79;
  generic64_t _var_80;
  generic64_t _var_81;
  generic64_t _var_82;
  generic64_t _var_83;
  generic64_t _var_84;
  generic64_t _var_85;
  generic64_t _var_86;
  generic64_t _var_87;
  generic64_t _var_88;
  generic64_t _var_89;
  generic64_t _var_90;
  generic64_t _var_91;
  generic8_t _var_92;
  generic32_t _var_93;
  generic64_t _var_94;
  generic64_t _var_95;
  generic8_t _var_96;
  generic32_t _var_97;
  generic64_t _var_98;
  generic32_t _var_99;
  generic32_t _var_100;
  generic8_t _var_101;
  generic64_t _var_102;
  generic64_t _var_103;
  generic64_t _var_104;
  generic64_t _var_105;
  generic64_t _var_106;
  generic64_t _var_107;
  generic64_t _var_108;
  generic64_t _var_109;
  generic64_t _var_110;
  generic64_t _var_111;
  generic64_t _var_112;
  generic64_t _var_113;
  generic64_t _var_114;
  generic64_t _var_115;
  generic64_t _var_116;
  generic64_t _var_117;
  generic64_t _var_118;
  generic64_t _var_119;
  generic64_t _var_120;
  generic8_t _var_121[8];
  generic8_t _var_122[8];
  generic8_t _var_123[8];
  generic8_t _var_124[8];
  generic8_t _var_125[8];
  generic8_t _var_126[8];
  generic8_t _var_127[4];
  generic8_t _var_128[4];
  generic8_t _var_129[4];
  generic8_t _var_130[8];
  generic8_t _var_131[8];
  generic8_t _var_132[8];
  generic8_t _var_133[8];
  generic8_t _var_134[8];
  generic8_t _var_135[8];
  generic8_t _var_136[8];
  generic8_t _var_137[4];
  generic8_t _var_138[4];
  generic8_t _var_139[8];
  generic8_t _var_140[8];
  generic8_t _var_141[8];
  generic8_t _var_142[8];
  generic8_t _var_143;
  generic8_t _var_144;
  generic8_t _var_145;
  generic8_t _var_146[8];
  generic8_t _var_147[8];
  generic8_t _var_148[8];
  generic8_t _var_149;
  generic8_t _var_150[8];
  generic8_t _var_151;
  generic8_t _var_152[8];
  generic8_t _var_153[8];
  generic8_t _var_154[8];
  generic8_t _var_155[8];
  generic8_t _var_156[8];
  generic8_t _var_157[8];
  generic8_t _var_158[8];
  generic8_t _var_159;
  generic8_t _var_160[8];
  generic8_t _var_161[4];
  generic8_t _var_162[8];
  generic8_t _var_163[8];
  generic8_t _var_164[8];
  generic8_t _var_165[8];
  generic8_t _var_166;
  generic8_t _var_167[8];
  generic8_t _var_168[8];
  generic8_t _var_169[8];
  generic8_t _var_170[8];
  generic8_t _var_171;
  generic8_t _var_172[8];
  generic8_t _var_173[4];
  generic8_t _var_174[4];
  generic8_t _var_175[8];
  generic8_t _var_176[8];
  generic8_t _var_177[8];
  generic8_t _var_178[8];
  generic8_t _var_179[8];
  generic8_t _var_180;
  generic8_t _var_181[8];
  generic8_t _var_182[8];
  generic8_t _var_183[8];
  generic8_t _var_184[8];
  generic8_t _var_185[8];
  generic8_t _var_186;
  generic8_t _var_187[8];
  generic8_t _var_188[8];
  generic8_t _var_189[8];
  generic8_t _var_190[4];
  generic8_t _var_191[8];
  generic8_t _var_192[4];
  generic8_t _var_193[4];
  generic8_t _var_194[8];
  generic8_t _var_195[4];
  generic8_t _var_196[4];
  generic8_t _var_197[8];
  generic8_t _var_198[8];
  generic8_t _var_199[4];
  generic8_t _var_200[8];
  generic8_t _var_201[8];
  generic8_t _var_202[8];
  generic8_t _var_203[8];
  generic8_t _var_204[8];
  generic8_t _var_205[8];
  generic8_t _var_206[8];
  generic8_t _var_207[8];
  generic8_t _var_208[8];
  generic8_t _var_209[8];
  generic8_t _var_210[8];
  generic8_t _var_211[8];
  generic8_t _var_212[8];
  generic8_t _var_213[8];
  generic8_t _var_214[8];
  generic8_t _var_215[8];
  generic8_t _var_216[8];
  generic8_t _var_217[8];
  generic8_t _var_218[8];
  generic8_t _var_219[8];
  generic8_t _var_220[8];
  generic8_t _var_221[8];
  generic8_t _var_222[8];
  generic8_t _var_223[8];
  generic8_t _var_224[8];
  generic8_t _var_225;
  generic8_t _var_226[4];
  generic8_t _var_227[8];
  generic8_t _var_228[8];
  generic8_t _var_229[8];
  generic8_t _var_230;
  ((generic64_t *)&_var_0)[2ul] = (generic64_t)ap;
  ((generic64_t *)&_var_0)[5ul] = (generic64_t)nl_arg;
  ((generic64_t *)&_var_0)[3ul] = (generic64_t)nl_type;
  ((generic32_t *)&_var_0)[2ul] = 0u;
  ((generic32_t *)&_var_0)[1ul] = 0u;
  ((generic32_t *)&_var_0)[12ul] = 0u;
  _var_119 = (generic64_t)fmt;
_label_0:
  _var_117 = _var_118;
  *(generic64_t *)&_var_135 = _var_119;
  *(generic64_t *)&_var_136 = _var_120;
  *(generic32_t *)&_var_137 = ((generic32_t *)&_var_0)[1ul];
  if ((generic8_t)((int32_t)*(generic32_t *)&_var_137 > 4294967295)) {
    *(generic32_t *)&_var_138 = ((generic32_t *)&_var_0)[2ul];
    _var_117 = (generic64_t)(uint64_t)(uint32_t)(2147483647u - *(generic32_t *)&_var_137);
    *(generic64_t *)&_var_139 = _var_117;
    ((generic32_t *)&_var_0)[1ul] = *(generic32_t *)&_var_137 + *(generic32_t *)&_var_138;
    if ((generic8_t)((int64_t)(*(generic64_t *)&_var_139 << 32ul) < (int64_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_138 << 32ul))) {
      *(int32_t **)&_var_121 = unreserved___errno_location();
      _var_117 = *(generic64_t *)&_var_121;
      *(generic64_t *)&_var_140 = _var_117;
      ((generic32_t *)&_var_0)[1ul] = 4294967295u;
      *(generic32_t *)*(generic64_t *)&_var_140 = 75u;
    } else {
    }
  } else {
  }
  if (*(generic8_t *)*(generic64_t *)&_var_135 == 0u) {
    if ((generic64_t)f == 0ul) {
      *(int8_t *)&_var_225 = ((generic32_t *)&_var_0)[12ul] == 0u;
      _var_1 = 0u;
      if (_var_225) {
        ((generic32_t *)&_var_0)[1ul] = _var_1;
        return (int32_t)((generic32_t *)&_var_0)[1ul];
      } else {
        _var_6 = ((generic64_t *)&_var_0)[3ul];
        *(generic32_t *)&_var_226 = ((generic32_t *)_var_6)[1ul];
        _var_5 = 1ul;
        if (*(generic32_t *)&_var_226 == 0u) {
          _var_4 = _var_5;
          *(generic64_t *)&_var_229 = _var_6 + (_var_4 << 2ul);
          _var_3 = 0ul;
        _label_1:
          _var_2 = 1u;
          if ((generic8_t)((uint64_t)_var_4 > 9ul)) {
            _var_1 = _var_2;
          } else {
            _var_4 = _var_4 + 1ul;
            *(int8_t *)&_var_230 = *(generic32_t *)(*(generic64_t *)&_var_229 + (_var_3 << 2ul)) == 0u;
            _var_3 = _var_3 + 1ul;
            _var_2 = 4294967295u;
            if (_var_230) {
              goto _label_1;
            } else {
            }
          }
        } else {
          _var_7 = 0ul;
          _var_8 = *(generic32_t *)&_var_226;
          _var_9 = 1ul;
        _label_2:
          *(generic64_t *)&_var_227 = _var_7;
          pop_arg((union arg *)((_var_9 << 4ul) + ((generic64_t *)&_var_0)[5ul]), (int32_t)_var_8, (va_list *)((generic64_t *)&_var_0)[2ul]);
          if (_var_9 == 9ul) {
            _var_1 = 1u;
          } else {
            _var_9 = _var_9 + 1ul;
            *(generic64_t *)&_var_228 = ((generic64_t *)&_var_0)[3ul];
            _var_8 = *(generic32_t *)((_var_9 << 2ul) + *(generic64_t *)&_var_228);
            _var_7 = *(generic64_t *)&_var_227 + 1ul;
            if (_var_8 == 0u) {
              _var_5 = (generic64_t)((int64_t)((*(generic64_t *)&_var_227 << 32ul) + 8589934592ul) >> 32l);
              _var_6 = *(generic64_t *)&_var_228;
            } else {
              goto _label_2;
            }
          }
        }
      }
    } else {
    }
  } else {
    _var_114 = 0ul;
    _var_115 = *(generic64_t *)&_var_135;
    _var_116 = _var_117;
  _label_3:
    *(generic64_t *)&_var_141 = _var_114;
    *(generic64_t *)&_var_142 = _var_115;
    _var_143 = *(generic8_t *)*(generic64_t *)&_var_142;
    _var_116 = _var_116 & 18446744073709551360ul | (generic64_t)(uint64_t)(uint8_t)_var_143;
    _var_115 = *(generic64_t *)&_var_142 + 1ul;
    _var_114 = *(generic64_t *)&_var_141 + 1ul;
    switch (_var_143) {
      case 37u: {
        *(int8_t *)&_var_144 = *(generic8_t *)*(generic64_t *)&_var_142 == 37u;
        _var_107 = *(generic64_t *)&_var_142;
        _var_108 = *(generic64_t *)&_var_142;
        if (_var_144) {
          _var_111 = 0ul;
          _var_112 = *(generic64_t *)&_var_142;
          _var_113 = *(generic64_t *)&_var_142;
        _label_4:
          _var_110 = _var_112;
          _var_109 = _var_113;
          if (*(generic8_t *)(*(generic64_t *)&_var_141 + (*(generic64_t *)&_var_135 + 1ul) + (_var_111 << 1ul)) == 37u) {
            _var_110 = *(generic64_t *)&_var_141 + (*(generic64_t *)&_var_135 + 2ul) + (_var_111 << 1ul);
            _var_109 = *(generic64_t *)&_var_141 + (*(generic64_t *)&_var_135 + 1ul) + _var_111;
            _var_113 = _var_113 + 1ul;
            _var_112 = _var_112 + 2ul;
            *(int8_t *)&_var_145 = *(generic8_t *)_var_110 == 37u;
            _var_111 = _var_111 + 1ul;
            if (_var_145) {
              goto _label_4;
            } else {
              _var_107 = _var_109;
              _var_108 = _var_110;
              *(generic64_t *)&_var_146 = _var_108;
              _var_106 = _var_107 - *(generic64_t *)&_var_135;
              ((generic32_t *)&_var_0)[2ul] = (generic32_t)_var_106;
              if ((generic64_t)f == 0ul) {
              } else {
                ((generic64_t *)&_var_0)[4ul] = _var_106;
                out(f, (const int8_t *)*(generic64_t *)&_var_135, (size_t)((int64_t)(_var_106 << 32ul) >> 32l));
                _var_106 = ((generic64_t *)&_var_0)[4ul];
              }
              *(generic64_t *)&_var_147 = _var_106;
              _var_11 = _var_116;
              _var_12 = *(generic64_t *)&_var_146;
              _var_13 = *(generic64_t *)&_var_136;
              if ((*(generic64_t *)&_var_147 & 4294967295ul) == 0ul) {
                _var_105 = *(generic64_t *)&_var_146 + 1ul;
                *(generic64_t *)&_var_148 = (generic64_t)(int64_t)(int8_t)*(generic8_t *)_var_105 + 4294967248ul & 4294967295ul;
                _var_104 = 4294967295ul;
                if ((generic8_t)((uint64_t)*(generic64_t *)&_var_148 > 9ul)) {
                } else {
                  *(int8_t *)&_var_149 = ((generic8_t *)*(generic64_t *)&_var_146)[2ul] == 36u;
                  _var_104 = 4294967295ul;
                  _var_105 = *(generic64_t *)&_var_146 + 1ul;
                  if (_var_149) {
                    ((generic32_t *)&_var_0)[12ul] = 1u;
                    _var_105 = *(generic64_t *)&_var_146 + 3ul;
                    _var_104 = *(generic64_t *)&_var_148;
                  } else {
                  }
                }
                *(generic64_t *)&_var_150 = _var_104;
                _var_91 = _var_105;
                _var_151 = *(generic8_t *)_var_91;
                _var_92 = _var_151;
                _var_93 = (generic32_t)(int32_t)(int8_t)_var_92;
                _var_90 = 0ul;
                if ((generic8_t)((uint32_t)(_var_93 + 4294967264u) > 31u)) {
                  *(generic64_t *)&_var_152 = _var_90;
                  *(generic64_t *)&_var_153 = _var_91;
                  if (_var_92 == 42u) {
                    *(generic64_t *)&_var_156 = (generic64_t)(int64_t)(int8_t)((generic8_t *)*(generic64_t *)&_var_153)[1ul];
                    if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_156 + 4294967248ul & 4294967294ul) > 9ul)) {
                      *(int8_t *)&_var_159 = ((generic32_t *)&_var_0)[12ul] == 0u;
                      _var_10 = 4294967295u;
                      if (_var_159) {
                        _var_81 = (generic64_t)(uint64_t)(uint32_t)_var_93;
                        _var_79 = 0ul;
                        _var_80 = *(generic64_t *)&_var_153 + 1ul;
                        if ((generic64_t)f == 0ul) {
                        } else {
                          *(generic64_t *)&_var_160 = ((generic64_t *)&_var_0)[2ul];
                          if ((generic8_t)((uint32_t)*(generic32_t *)*(generic64_t *)&_var_160 > 47u)) {
                            _var_82 = ((generic64_t *)*(generic64_t *)&_var_160)[1ul];
                            ((generic64_t *)*(generic64_t *)&_var_160)[1ul] = _var_82 + 8ul;
                            _var_83 = *(generic64_t *)&_var_153 + 1ul;
                            _var_84 = *(generic64_t *)&_var_160;
                          } else {
                            *(generic32_t *)&_var_161 = *(generic32_t *)*(generic64_t *)&_var_160 + 8u;
                            _var_82 = ((generic64_t *)*(generic64_t *)&_var_160)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)*(generic64_t *)&_var_160;
                            *(generic32_t *)*(generic64_t *)&_var_160 = *(generic32_t *)&_var_161;
                            _var_83 = *(generic64_t *)&_var_153 + 1ul;
                            _var_84 = _var_81;
                          }
                          _var_80 = _var_83;
                          _var_81 = _var_84;
                          _var_79 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)_var_82;
                        }
                        _var_76 = _var_79;
                        _var_77 = _var_80;
                        _var_78 = _var_81;
                        _var_75 = *(generic64_t *)&_var_152;
                        if ((generic8_t)((uint64_t)_var_76 < 2147483648ul)) {
                        } else {
                          _var_75 = *(generic64_t *)&_var_152 & 4294959103ul | 8192ul;
                          _var_76 = 0ul - _var_79 & 4294967295ul;
                          _var_77 = _var_80;
                          _var_78 = _var_81;
                        }
                        *(generic64_t *)&_var_162 = _var_75;
                        *(generic64_t *)&_var_163 = _var_76;
                        _var_66 = _var_77;
                        *(generic64_t *)&_var_165 = _var_78;
                        _var_67 = *(generic64_t *)&_var_165;
                        *(int8_t *)&_var_166 = *(generic8_t *)_var_66 == 46u;
                        _var_65 = 4294967295u;
                        if (_var_166) {
                          if (((generic8_t *)_var_77)[1ul] == 42u) {
                            *(void **)&_var_164 = (void *)(_var_77 + 2ul);
                            *(generic64_t *)&_var_169 = (generic64_t)(int64_t)(int8_t)*(generic8_t *)*(void **)&_var_164;
                            if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_169 + 4294967248ul & 4294967294ul) > 9ul)) {
                              *(int8_t *)&_var_171 = ((generic32_t *)&_var_0)[12ul] == 0u;
                              _var_10 = 4294967295u;
                              if (_var_171) {
                                _var_65 = 0u;
                                _var_66 = _var_77 + 2ul;
                                _var_67 = *(generic64_t *)&_var_165;
                                if ((generic64_t)f == 0ul) {
                                } else {
                                  *(generic64_t *)&_var_172 = ((generic64_t *)&_var_0)[2ul];
                                  if ((generic8_t)((uint32_t)*(generic32_t *)*(generic64_t *)&_var_172 > 47u)) {
                                    _var_69 = ((generic64_t *)*(generic64_t *)&_var_172)[1ul];
                                    ((generic64_t *)*(generic64_t *)&_var_172)[1ul] = _var_69 + 8ul;
                                    _var_68 = *(generic64_t *)&_var_172;
                                  } else {
                                    *(generic32_t *)&_var_173 = *(generic32_t *)*(generic64_t *)&_var_172 + 8u;
                                    _var_69 = ((generic64_t *)*(generic64_t *)&_var_172)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)*(generic64_t *)&_var_172;
                                    *(generic32_t *)*(generic64_t *)&_var_172 = *(generic32_t *)&_var_173;
                                    _var_68 = *(generic64_t *)&_var_165;
                                  }
                                  _var_67 = _var_68;
                                  _var_65 = *(generic32_t *)_var_69;
                                }
                                *(generic32_t *)&_var_174 = _var_65;
                                _var_63 = _var_66;
                                *(generic64_t *)&_var_175 = _var_63 + 1ul;
                                _var_62 = 0ul;
                                _var_64 = 0ul;
                              _label_5:
                                *(generic64_t *)&_var_178 = _var_64;
                                *(void **)&_var_177 = (void *)_var_63;
                                if ((generic8_t)((uint32_t)((generic32_t)(int32_t)(int8_t)*(generic8_t *)*(void **)&_var_177 + 4294967231u) > 57u)) {
                                  _var_1 = 4294967295u;
                                } else {
                                  *(generic64_t *)&_var_176 = *(generic64_t *)&_var_175 + _var_62;
                                  *(generic64_t *)&_var_179 = (generic64_t)(int64_t)(int32_t)((generic32_t)(int32_t)(int8_t)*(generic8_t *)*(void **)&_var_177 + 4294967231u);
                                  _var_63 = _var_63 + 1ul;
                                  _var_180 = ((generic8_t *)(*(generic64_t *)&_var_178 * 58ul + *(generic64_t *)&_var_179))[4215456ul];
                                  _var_64 = (generic64_t)(uint64_t)(uint8_t)_var_180;
                                  _var_62 = _var_62 + 1ul;
                                  if ((generic8_t)((uint32_t)((generic32_t)(uint32_t)(uint8_t)_var_180 + 4294967295u) < 8u)) {
                                    goto _label_5;
                                  } else {
                                    _var_10 = 4294967295u;
                                    if (_var_180 == 0u) {
                                      _var_1 = _var_10;
                                    } else if (_var_180 == 21u) {
                                      _var_10 = 4294967295u;
                                      _var_59 = *(generic64_t *)&_var_150;
                                      _var_60 = _var_67;
                                      _var_61 = *(generic64_t *)&_var_136;
                                      if ((generic8_t)((uint64_t)*(generic64_t *)&_var_150 < 2147483648ul)) {
                                      } else {
                                        _var_11 = _var_59;
                                        _var_57 = _var_60;
                                        _var_58 = _var_61;
                                        _var_13 = _var_58;
                                        _var_12 = *(generic64_t *)&_var_176;
                                        _var_56 = *(generic64_t *)&_var_147;
                                        if ((generic64_t)f == 0ul) {
                                          goto _label_0;
                                        } else {
                                          *(generic64_t *)&_var_184 = _var_56;
                                          *(generic64_t *)&_var_185 = _var_58;
                                          _var_186 = *(generic8_t *)*(void **)&_var_177;
                                          _var_55 = (generic64_t)(uint64_t)(uint32_t)(int32_t)(int8_t)_var_186;
                                          _var_54 = 0ul;
                                          if (*(generic64_t *)&_var_178 == 0ul) {
                                            _var_53 = _var_55;
                                            *(generic64_t *)&_var_126 = lshift(_var_54, 4294967272u);
                                          } else if ((generic8_t)((generic64_t)(uint64_t)(uint8_t)(_var_186 & 15u) + 18446744073709551613ul) == 0u) {
                                            _var_55 = (generic64_t)(uint64_t)(uint32_t)((generic32_t)(int32_t)(int8_t)_var_186 & 4294967263u);
                                            _var_54 = _var_55;
                                          } else {
                                            *(generic64_t *)&_var_124 = lshift((generic64_t)(uint64_t)(uint8_t)(_var_186 & 15u) + 18446744073709551613ul & 255ul, 0u);
                                            *(generic64_t *)&_var_125 = lshift(((generic64_t)(uint64_t)(uint8_t)(_var_186 & 15u) + 18446744073709551613ul ^ (generic64_t)(uint64_t)(uint8_t)((generic8_t)((generic64_t)(uint64_t)(uint8_t)(_var_186 & 15u) + 18446744073709551613ul) + 3u)) & (generic64_t)(uint64_t)(uint8_t)((generic8_t)((generic64_t)(uint64_t)(uint8_t)(_var_186 & 15u) + 18446744073709551613ul) + 3u ^ 3u), 4u);
                                            _var_53 = (generic64_t)(uint64_t)(uint32_t)(int32_t)(int8_t)_var_186;
                                          }
                                          *(generic64_t *)&_var_187 = _var_53;
                                          _var_15 = (*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul;
                                          _var_14 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_174;
                                          _var_16 = 4215957ul;
                                          _var_17 = (generic64_t)&_var_0 + 144ul;
                                          _var_18 = *(generic64_t *)&_var_135;
                                          if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_187 + 4294967231ul & 4294967295ul) > 55ul)) {
                                            *(generic64_t *)&_var_220 = _var_14;
                                            *(generic64_t *)&_var_221 = _var_15;
                                            *(generic64_t *)&_var_222 = _var_16;
                                            *(generic64_t *)&_var_223 = _var_18;
                                            ((generic64_t *)&_var_0)[4ul] = _var_17 - *(generic64_t *)&_var_223;
                                            _var_11 = (generic64_t)(uint64_t)(uint32_t)((generic32_t *)&_var_0)[2ul];
                                            *(generic64_t *)&_var_224 = _var_11;
                                            ((generic64_t *)&_var_0)[7ul] = *(generic64_t *)&_var_222;
                                            ((generic32_t *)&_var_0)[13ul] = (generic32_t)((((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224);
                                            pad(f, 32, (int32_t)(generic32_t)((generic8_t)((int64_t)((((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224 << 32ul) < (int64_t)(*(generic64_t *)&_var_163 << 32ul)) ? *(generic64_t *)&_var_163 : (((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224), (int32_t)(generic32_t)((((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224), (int32_t)(generic32_t)*(generic64_t *)&_var_221);
                                            out(f, (const int8_t *)((generic64_t *)&_var_0)[7ul], (size_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[2ul]);
                                            ((generic32_t *)&_var_0)[2ul] = ((generic32_t *)&_var_0)[13ul];
                                            pad(f, 48, (int32_t)(generic32_t)((generic8_t)((int64_t)((((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224 << 32ul) < (int64_t)(*(generic64_t *)&_var_163 << 32ul)) ? *(generic64_t *)&_var_163 : (((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224), (int32_t)((generic32_t *)&_var_0)[13ul], (int32_t)((generic32_t)*(generic64_t *)&_var_221 ^ 65536u));
                                            pad(f, 48, (int32_t)(generic32_t)((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220), (int32_t)((generic32_t *)&_var_0)[8ul], 0);
                                            out(f, (const int8_t *)*(generic64_t *)&_var_223, (size_t)((generic64_t *)&_var_0)[4ul]);
                                            pad(f, 32, (int32_t)(generic32_t)((generic8_t)((int64_t)((((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224 << 32ul) < (int64_t)(*(generic64_t *)&_var_163 << 32ul)) ? *(generic64_t *)&_var_163 : (((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224), (int32_t)((generic32_t *)&_var_0)[2ul], (int32_t)((generic32_t)*(generic64_t *)&_var_221 ^ 8192u));
                                            ((generic32_t *)&_var_0)[2ul] = (generic32_t)((generic8_t)((int64_t)((((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224 << 32ul) < (int64_t)(*(generic64_t *)&_var_163 << 32ul)) ? *(generic64_t *)&_var_163 : (((generic8_t)((int64_t)(*(generic64_t *)&_var_220 << 32ul) >> 32l < (int64_t)((generic64_t *)&_var_0)[4ul]) ? ((generic64_t *)&_var_0)[4ul] : *(generic64_t *)&_var_220) & 4294967295ul) + *(generic64_t *)&_var_224);
                                            _var_12 = *(generic64_t *)&_var_176;
                                            _var_13 = *(generic64_t *)&_var_185;
                                          } else {
                                            *(generic64_t *)&_var_188 = ((generic64_t *)((*(generic64_t *)&_var_187 + 4294967231ul & 4294967295ul) << 3ul))[526864ul];
                                            _var_14 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_174;
                                            _var_15 = (*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul;
                                            _var_16 = 4215957ul;
                                            _var_17 = (generic64_t)&_var_0 + 144ul;
                                            _var_18 = *(generic64_t *)&_var_135;
                                            _var_32 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_174;
                                            _var_33 = (*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul;
                                            _var_34 = *(generic64_t *)&_var_187;
                                            _var_51 = *(generic32_t *)&_var_174;
                                            _var_52 = *(generic64_t *)&_var_187 + 4294967231ul & 4294967295ul;
                                            switch (*(generic64_t *)&_var_188) {
                                              case 4206106ul: {
                                                _var_11 = *(generic64_t *)&_var_187 + 4294967231ul & 4294967295ul;
                                                _var_12 = *(generic64_t *)&_var_176;
                                                _var_13 = *(generic64_t *)&_var_185;
                                                if ((generic8_t)((uint64_t)*(generic64_t *)&_var_178 < 8ul)) {
                                                  *(generic64_t *)&_var_219 = ((generic64_t *)(*(generic64_t *)&_var_178 << 3ul))[526920ul];
                                                  _var_11 = *(generic64_t *)&_var_187 + 4294967231ul & 4294967295ul;
                                                  _var_12 = *(generic64_t *)&_var_176;
                                                  _var_13 = *(generic64_t *)&_var_185;
                                                  switch (*(generic64_t *)&_var_219) {
                                                    case 4205394ul: {
                                                    }
                                                    case 4206124ul: {
                                                      _var_11 = ((generic64_t *)&_var_0)[10ul];
                                                      *(generic32_t *)_var_11 = ((generic32_t *)&_var_0)[1ul];
                                                      _var_12 = *(generic64_t *)&_var_176;
                                                      _var_13 = *(generic64_t *)&_var_185;
                                                    }
                                                    case 4206140ul: {
                                                      _var_11 = ((generic64_t *)&_var_0)[10ul];
                                                      *(generic16_t *)_var_11 = (generic16_t)((generic32_t *)&_var_0)[1ul];
                                                      _var_12 = *(generic64_t *)&_var_176;
                                                      _var_13 = *(generic64_t *)&_var_185;
                                                    }
                                                    case 4206157ul: {
                                                      _var_11 = ((generic64_t *)&_var_0)[10ul];
                                                      *(generic8_t *)_var_11 = ((generic8_t *)&_var_0)[4ul];
                                                      _var_12 = *(generic64_t *)&_var_176;
                                                      _var_13 = *(generic64_t *)&_var_185;
                                                    }
                                                    case 4206175ul: {
                                                      _var_11 = (generic64_t)(int64_t)(int32_t)((generic32_t *)&_var_0)[1ul];
                                                      *(generic64_t *)((generic64_t *)&_var_0)[10ul] = _var_11;
                                                      _var_12 = *(generic64_t *)&_var_176;
                                                      _var_13 = *(generic64_t *)&_var_185;
                                                    }
                                                    default: {
                                                    }
                                                  }
                                                } else {
                                                }
                                              }
                                              case 4206193ul: {
                                                _var_32 = (generic64_t)(uint64_t)(uint32_t)llvm.umax.i32(*(generic32_t *)&_var_174, 16u);
                                                _var_33 = ((*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul) & 4294967287ul | 8ul;
                                                _var_34 = 120ul;
                                                *(generic64_t *)&_var_207 = _var_32;
                                                *(generic64_t *)&_var_208 = _var_33;
                                                *(generic64_t *)&_var_209 = _var_34;
                                                *(generic64_t *)&_var_210 = ((generic64_t *)&_var_0)[10ul];
                                                _var_28 = (generic64_t)&_var_0 + 144ul;
                                                if (*(generic64_t *)&_var_210 == 0ul) {
                                                  _var_27 = _var_28;
                                                  _var_25 = *(generic64_t *)&_var_207;
                                                  _var_26 = *(generic64_t *)&_var_208;
                                                  if (*(generic64_t *)&_var_210 == 0ul ? 1u : (generic8_t)((*(generic64_t *)&_var_208 & 8ul) == 0ul)) {
                                                    _var_21 = _var_25;
                                                    _var_22 = _var_26;
                                                    _var_24 = _var_27;
                                                    _var_20 = *(generic64_t *)&_var_184 & 4294967295ul;
                                                    _var_19 = *(generic64_t *)&_var_184;
                                                    _var_23 = 4215957ul;
                                                  } else {
                                                    _var_23 = (generic64_t)(int64_t)(int32_t)((generic32_t)((int32_t)(generic32_t)*(generic64_t *)&_var_209 >> 4) + 4215957u);
                                                    _var_19 = *(generic64_t *)&_var_184;
                                                    _var_20 = 2ul;
                                                    _var_21 = *(generic64_t *)&_var_207;
                                                    _var_22 = *(generic64_t *)&_var_208;
                                                    _var_24 = _var_28;
                                                  }
                                                  *(generic64_t *)&_var_213 = _var_19;
                                                  *(generic64_t *)&_var_216 = _var_23;
                                                  *(generic64_t *)&_var_217 = _var_24;
                                                  *(generic64_t *)&_var_214 = (generic8_t)((int32_t)(generic32_t)_var_21 < 0) ? _var_22 : _var_22 & 4294901759ul;
                                                  if ((generic32_t)_var_21 == 0u ? (generic8_t)(((generic64_t *)&_var_0)[10ul] == 0ul) : 0u) {
                                                    ((generic32_t *)&_var_0)[2ul] = (generic32_t)_var_20;
                                                    _var_14 = *(generic64_t *)&_var_213 & 4294967295ul;
                                                    _var_15 = *(generic64_t *)&_var_214;
                                                    _var_16 = *(generic64_t *)&_var_216;
                                                    _var_17 = (generic64_t)&_var_0 + 144ul;
                                                    _var_18 = (generic64_t)&_var_0 + 144ul;
                                                  } else {
                                                    *(generic64_t *)&_var_218 = (generic64_t)(uint64_t)(uint8_t)(((generic64_t *)&_var_0)[10ul] == 0ul);
                                                    *(generic64_t *)&_var_215 = (generic64_t)((int64_t)(_var_21 << 32ul) >> 32l);
                                                    ((generic32_t *)&_var_0)[2ul] = (generic32_t)_var_20;
                                                    _var_14 = llvm.smax.i64((generic64_t)&_var_0 + 144ul - *(generic64_t *)&_var_217 + *(generic64_t *)&_var_218, *(generic64_t *)&_var_215);
                                                    _var_15 = *(generic64_t *)&_var_214;
                                                    _var_16 = *(generic64_t *)&_var_216;
                                                    _var_17 = (generic64_t)&_var_0 + 144ul;
                                                    _var_18 = *(generic64_t *)&_var_217;
                                                  }
                                                } else {
                                                  _var_29 = 0ul;
                                                  _var_30 = _var_57;
                                                  _var_31 = *(generic64_t *)&_var_210;
                                                _label_6:
                                                  *(generic64_t *)&_var_211 = _var_29;
                                                  *(generic64_t *)&_var_212 = _var_31;
                                                  _var_31 = (generic64_t)((uint64_t)*(generic64_t *)&_var_212 >> 4ul);
                                                  _var_30 = _var_30 & 4294967040ul | (generic64_t)(uint64_t)(uint8_t)*(generic8_t *)(*(generic64_t *)&_var_212 & 15ul | 4215424ul) | *(generic64_t *)&_var_209 & 32ul;
                                                  *(generic8_t *)((generic64_t)&_var_0 + 143ul - *(generic64_t *)&_var_211) = (generic8_t)_var_30;
                                                  _var_29 = *(generic64_t *)&_var_211 + 1ul;
                                                  if ((generic8_t)((uint64_t)*(generic64_t *)&_var_212 < 16ul)) {
                                                    _var_28 = (generic64_t)&_var_0 + 143ul - *(generic64_t *)&_var_211;
                                                  } else {
                                                    goto _label_6;
                                                  }
                                                }
                                              }
                                              case 4206216ul: {
                                              }
                                              case 4206315ul: {
                                                *(generic64_t *)&_var_204 = ((generic64_t *)&_var_0)[10ul];
                                                _var_35 = (generic64_t)&_var_0 + 144ul;
                                                if (*(generic64_t *)&_var_204 == 0ul) {
                                                  _var_27 = _var_35;
                                                  _var_25 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_174;
                                                  _var_26 = (*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul;
                                                  if ((((*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul) & 8ul) == 0ul) {
                                                  } else {
                                                    _var_25 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_174;
                                                    _var_26 = (*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul;
                                                    _var_27 = _var_35;
                                                    if ((generic8_t)((int64_t)((generic64_t)&_var_0 + 144ul - _var_35) < (int64_t)(int32_t)*(generic32_t *)&_var_174)) {
                                                    } else {
                                                      _var_25 = (generic64_t)&_var_0 + 144ul - _var_35 + 1ul & 4294967295ul;
                                                      _var_26 = (*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul;
                                                      _var_27 = _var_35;
                                                    }
                                                  }
                                                } else {
                                                  _var_36 = 0ul;
                                                  _var_37 = *(generic64_t *)&_var_204;
                                                _label_7:
                                                  *(generic64_t *)&_var_205 = _var_36;
                                                  *(generic64_t *)&_var_206 = _var_37;
                                                  _var_37 = (generic64_t)((uint64_t)*(generic64_t *)&_var_206 >> 3ul);
                                                  *(generic8_t *)((generic64_t)&_var_0 + 143ul - *(generic64_t *)&_var_205) = (generic8_t)*(generic64_t *)&_var_206 & 7u | 48u;
                                                  _var_36 = *(generic64_t *)&_var_205 + 1ul;
                                                  if ((generic8_t)((uint64_t)*(generic64_t *)&_var_206 < 8ul)) {
                                                    _var_35 = (generic64_t)&_var_0 + 143ul - *(generic64_t *)&_var_205;
                                                  } else {
                                                    goto _label_7;
                                                  }
                                                }
                                              }
                                              case 4206395ul: {
                                                if ((generic8_t)((int64_t)((generic64_t *)&_var_0)[10ul] > -1l)) {
                                                  *(generic64_t *)&_var_133 = lshift(((generic64_t *)&_var_0)[10ul], 4294967240u);
                                                  _var_38 = 1ul;
                                                  _var_39 = 4215958ul;
                                                  if ((((*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul) & 2048ul) == 0ul) {
                                                    _var_38 = (((*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul) & 1ul) == 0ul ? *(generic64_t *)&_var_184 & 4294967295ul : 1ul;
                                                    _var_39 = (((*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul) & 1ul) == 0ul ? 4215957ul : 4215959ul;
                                                  } else {
                                                  }
                                                } else {
                                                  ((generic64_t *)&_var_0)[10ul] = 0ul - ((generic64_t *)&_var_0)[10ul];
                                                  _var_38 = 1ul;
                                                  _var_39 = 4215957ul;
                                                }
                                                _var_20 = _var_38;
                                                *(generic64_t *)&_var_202 = _var_39;
                                                *(generic64_t *)&_var_203 = ((generic64_t *)&_var_0)[10ul];
                                                ((generic64_t *)&_var_0)[4ul] = *(generic64_t *)&_var_184;
                                                ((generic64_t *)&_var_0)[1ul] = *(generic64_t *)&_var_202;
                                                *(int8_t **)&_var_134 = fmt_u((unreserved_uintmax_t)*(generic64_t *)&_var_203, (int8_t *)&_var_0 + 144ul);
                                                _var_24 = *(generic64_t *)&_var_134;
                                                _var_23 = ((generic64_t *)&_var_0)[1ul];
                                                _var_19 = ((generic64_t *)&_var_0)[4ul];
                                                _var_21 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_174;
                                                _var_22 = (*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul;
                                              }
                                              case 4206461ul: {
                                                _var_38 = *(generic64_t *)&_var_184 & 4294967295ul;
                                                _var_39 = 4215957ul;
                                              }
                                              case 4206607ul: {
                                                ((generic8_t *)&_var_0)[143ul] = (generic8_t)((generic64_t *)&_var_0)[10ul];
                                                _var_14 = 1ul;
                                                _var_15 = *(generic64_t *)&_var_162 & 4294901759ul;
                                                _var_16 = 4215957ul;
                                                _var_17 = (generic64_t)&_var_0 + 144ul;
                                                _var_18 = (generic64_t)&_var_0 + 143ul;
                                              }
                                              case 4206659ul: {
                                                *(int32_t **)&_var_130 = unreserved___errno_location();
                                                *(int8_t **)&_var_131 = strerror((int32_t)*(generic32_t *)*(generic64_t *)&_var_130);
                                                _var_40 = *(generic64_t *)&_var_131;
                                                *(generic64_t *)&_var_201 = _var_40;
                                                ((generic64_t *)&_var_0)[4ul] = (generic64_t)(int64_t)(int32_t)*(generic32_t *)&_var_174;
                                                *(void **)&_var_132 = memchr((const void *)*(generic64_t *)&_var_201, 0, (size_t)(int64_t)(int32_t)*(generic32_t *)&_var_174);
                                                if (*(generic64_t *)&_var_132 == 0ul) {
                                                  _var_17 = ((generic64_t *)&_var_0)[4ul] + *(generic64_t *)&_var_201;
                                                  _var_14 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_174;
                                                  _var_15 = *(generic64_t *)&_var_162 & 4294901759ul;
                                                  _var_16 = 4215957ul;
                                                  _var_18 = *(generic64_t *)&_var_201;
                                                } else {
                                                  _var_14 = *(generic64_t *)&_var_132 - *(generic64_t *)&_var_201 & 4294967295ul;
                                                  _var_15 = *(generic64_t *)&_var_162 & 4294901759ul;
                                                  _var_16 = 4215957ul;
                                                  _var_17 = *(generic64_t *)&_var_132;
                                                  _var_18 = *(generic64_t *)&_var_201;
                                                }
                                              }
                                              case 4206676ul:
                                                _var_40 = ((generic64_t *)&_var_0)[10ul] == 0ul ? 4215967ul : ((generic64_t *)&_var_0)[10ul];
                                              case 4206750ul: {
                                                *(generic64_t *)&_var_189 = ((generic64_t *)&_var_0)[10ul];
                                                ((generic32_t *)&_var_0)[19ul] = 0u;
                                                ((generic32_t *)&_var_0)[18ul] = (generic32_t)*(generic64_t *)&_var_189;
                                                ((generic64_t *)&_var_0)[10ul] = (generic64_t)&_var_0 + 72ul;
                                                _var_51 = 4294967295u;
                                                _var_52 = (generic64_t)&_var_0 + 72ul;
                                                *(generic32_t *)&_var_190 = _var_51;
                                                _var_45 = _var_52;
                                                _var_44 = *(generic64_t *)&_var_184;
                                                if ((generic8_t)((uint32_t)((generic32_t)*(generic64_t *)&_var_184 - *(generic32_t *)&_var_190) > (uint32_t)(*(generic32_t *)&_var_190 ^ 4294967295u))) {
                                                  *(generic64_t *)&_var_191 = ((generic64_t *)&_var_0)[10ul];
                                                  *(generic32_t *)&_var_192 = *(generic32_t *)*(generic64_t *)&_var_191;
                                                  _var_44 = *(generic64_t *)&_var_184;
                                                  _var_45 = _var_52;
                                                  if (*(generic32_t *)&_var_192 == 0u) {
                                                    *(generic64_t *)&_var_197 = _var_44;
                                                    _var_41 = _var_45;
                                                    pad(f, 32, (int32_t)(generic32_t)*(generic64_t *)&_var_163, (int32_t)(generic32_t)*(generic64_t *)&_var_197, (int32_t)(generic32_t)((*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul));
                                                    if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_197 & 4294967295ul) > (uint64_t)(uint32_t)((generic32_t *)&_var_0)[2ul])) {
                                                      *(generic64_t *)&_var_198 = ((generic64_t *)&_var_0)[10ul];
                                                      *(generic32_t *)&_var_199 = *(generic32_t *)*(generic64_t *)&_var_198;
                                                      _var_41 = _var_45;
                                                      if (*(generic32_t *)&_var_199 == 0u) {
                                                        _var_11 = _var_41;
                                                        pad(f, 32, (int32_t)(generic32_t)*(generic64_t *)&_var_163, (int32_t)(generic32_t)*(generic64_t *)&_var_197, (int32_t)((generic32_t)((*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul) ^ 8192u));
                                                        ((generic32_t *)&_var_0)[2ul] = (generic32_t)((generic8_t)((int64_t)(*(generic64_t *)&_var_163 << 32ul) < (int64_t)(*(generic64_t *)&_var_197 << 32ul)) ? *(generic64_t *)&_var_197 : *(generic64_t *)&_var_163);
                                                        _var_12 = *(generic64_t *)&_var_176;
                                                        _var_13 = *(generic64_t *)&_var_185;
                                                      } else {
                                                        _var_42 = 0ul;
                                                        _var_43 = *(generic32_t *)&_var_199;
                                                      _label_8:
                                                        *(generic64_t *)&_var_200 = _var_42;
                                                        *(int32_t *)&_var_129 = wctomb((int8_t *)&_var_0 + 68ul, (wchar_t)_var_43);
                                                        ((generic32_t *)&_var_0)[2ul] = ((generic32_t *)&_var_0)[2ul] + *(generic32_t *)&_var_129;
                                                        if ((generic8_t)((int64_t)(*(generic64_t *)&_var_197 << 32ul) >> 32l < (int64_t)(int32_t)((generic32_t *)&_var_0)[2ul])) {
                                                          _var_41 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_129;
                                                        } else {
                                                          out(f, (const int8_t *)&_var_0 + 68ul, (size_t)(int64_t)(int32_t)*(generic32_t *)&_var_129);
                                                          if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_197 & 4294967295ul) > (uint64_t)(uint32_t)((generic32_t *)&_var_0)[2ul])) {
                                                            _var_43 = *(generic32_t *)(*(generic64_t *)&_var_198 + 4ul + (*(generic64_t *)&_var_200 << 2ul));
                                                            _var_42 = *(generic64_t *)&_var_200 + 1ul;
                                                            if (_var_43 == 0u) {
                                                            } else {
                                                              goto _label_8;
                                                            }
                                                          } else {
                                                          }
                                                        }
                                                      }
                                                    } else {
                                                    }
                                                  } else {
                                                    _var_47 = *(generic32_t *)&_var_192;
                                                    _var_48 = *(generic64_t *)&_var_191;
                                                    _var_49 = *(generic64_t *)&_var_184;
                                                    _var_50 = (generic32_t)*(generic64_t *)&_var_184;
                                                  _label_9:
                                                    *(generic32_t *)&_var_193 = _var_47;
                                                    *(generic64_t *)&_var_194 = _var_49;
                                                    *(generic32_t *)&_var_195 = _var_50;
                                                    ((generic64_t *)&_var_0)[4ul] = _var_48 + 4ul;
                                                    *(int32_t *)&_var_128 = wctomb((int8_t *)&_var_0 + 68ul, (wchar_t)*(generic32_t *)&_var_193);
                                                    if ((generic8_t)((int32_t)*(generic32_t *)&_var_128 > 4294967295)) {
                                                      _var_46 = *(generic64_t *)&_var_194;
                                                      if ((generic8_t)((uint32_t)(*(generic32_t *)&_var_190 - (*(generic32_t *)&_var_195 + *(generic32_t *)&_var_128)) > (uint32_t)(*(generic32_t *)&_var_128 ^ 4294967295u))) {
                                                        _var_44 = _var_46;
                                                        _var_45 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_128;
                                                      } else {
                                                        _var_46 = (*(generic64_t *)&_var_194 & 4294967295ul) + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_128;
                                                        *(generic32_t *)&_var_196 = (generic32_t)_var_46;
                                                        if ((generic8_t)((uint32_t)(*(generic32_t *)&_var_196 - *(generic32_t *)&_var_190) > (uint32_t)(*(generic32_t *)&_var_190 ^ 4294967295u))) {
                                                          _var_48 = ((generic64_t *)&_var_0)[4ul];
                                                          _var_47 = *(generic32_t *)_var_48;
                                                          _var_46 = (*(generic64_t *)&_var_194 & 4294967295ul) + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_128;
                                                          _var_49 = (*(generic64_t *)&_var_194 & 4294967295ul) + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_128;
                                                          _var_50 = *(generic32_t *)&_var_196;
                                                          if (_var_47 == 0u) {
                                                          } else {
                                                            goto _label_9;
                                                          }
                                                        } else {
                                                        }
                                                      }
                                                    } else {
                                                      _var_1 = 4294967295u;
                                                    }
                                                  }
                                                } else {
                                                }
                                              }
                                              case 4206781ul: {
                                              }
                                              case 4206857ul: {
                                                ((generic64_t *)&_var_0)[2305843009213693951ul] = ((generic64_t *)&_var_0)[11ul];
                                                ((generic64_t *)&_var_0)[2305843009213693950ul] = ((generic64_t *)&_var_0)[10ul];
                                                *(int32_t *)&_var_127 = fmt_fp(f, (float128_t)(uint128_t)(uint64_t)*(generic64_t *)&_var_185, (int32_t)(generic32_t)*(generic64_t *)&_var_163, (int32_t)*(generic32_t *)&_var_174, (int32_t)(generic32_t)((*(generic64_t *)&_var_162 & 8192ul) == 0ul ? *(generic64_t *)&_var_162 : *(generic64_t *)&_var_162 & 4294901759ul), (int32_t)(generic32_t)*(generic64_t *)&_var_187);
                                                ((generic32_t *)&_var_0)[2ul] = *(generic32_t *)&_var_127;
                                                _var_11 = ((generic64_t *)&_var_0)[2305843009213693950ul];
                                                _var_12 = *(generic64_t *)&_var_176;
                                                _var_13 = *(generic64_t *)&_var_185;
                                              }
                                              case 4206893ul: {
                                              }
                                              default: {
                                              }
                                            }
                                          }
                                        }
                                      }
                                    } else if ((generic8_t)((uint64_t)*(generic64_t *)&_var_150 < 2147483648ul)) {
                                      _var_60 = ((generic64_t *)&_var_0)[3ul];
                                      *(generic64_t *)&_var_182 = ((generic64_t *)&_var_0)[5ul];
                                      *(generic32_t *)((generic64_t)((int64_t)(*(generic64_t *)&_var_150 << 32ul) >> 30l) + _var_60) = (generic32_t)(uint32_t)(uint8_t)_var_180;
                                      _var_59 = (generic64_t)((int64_t)(*(generic64_t *)&_var_150 << 32ul) >> 28l);
                                      _var_61 = *(generic64_t *)(_var_59 + *(generic64_t *)&_var_182);
                                      *(generic64_t *)&_var_183 = ((generic64_t *)(_var_59 + *(generic64_t *)&_var_182))[1ul];
                                      ((generic64_t *)&_var_0)[10ul] = _var_61;
                                      ((generic64_t *)&_var_0)[11ul] = *(generic64_t *)&_var_183;
                                    } else {
                                      _var_10 = 0u;
                                      if ((generic64_t)f == 0ul) {
                                      } else {
                                        *(generic64_t *)&_var_181 = ((generic64_t *)&_var_0)[2ul];
                                        ((generic64_t *)&_var_0)[4ul] = *(generic64_t *)&_var_147;
                                        pop_arg((union arg *)&_var_0 + 5ul, (int32_t)(uint32_t)(uint8_t)_var_180, (va_list *)*(generic64_t *)&_var_181);
                                        _var_56 = ((generic64_t *)&_var_0)[4ul];
                                        _var_57 = (generic64_t)&_var_0 + 80ul;
                                        _var_58 = *(generic64_t *)&_var_136;
                                      }
                                    }
                                  }
                                }
                              } else {
                              }
                            } else if (((generic8_t *)_var_77)[3ul] == 36u) {
                              _var_67 = ((generic64_t *)&_var_0)[3ul];
                              *(generic64_t *)&_var_170 = ((generic64_t *)&_var_0)[5ul];
                              _var_66 = _var_77 + 4ul;
                              ((generic32_t *)((*(generic64_t *)&_var_169 << 2ul) + _var_67))[4611686018427387856ul] = 10u;
                              _var_65 = ((generic32_t *)(((generic64_t)(int64_t)(int8_t)*(generic8_t *)*(void **)&_var_164 << 4ul) + *(generic64_t *)&_var_170))[4611686018427387712ul];
                            } else {
                            }
                          } else {
                            *(generic64_t *)&_var_167 = (generic64_t)(int64_t)(int8_t)((generic8_t *)_var_77)[1ul] + 4294967248ul & 4294967295ul;
                            _var_70 = *(generic64_t *)&_var_147;
                            _var_71 = _var_77 + 1ul;
                            if ((generic8_t)((uint64_t)*(generic64_t *)&_var_167 > 9ul)) {
                              _var_66 = _var_71;
                              _var_65 = (generic32_t)_var_70;
                              _var_67 = *(generic64_t *)&_var_165;
                            } else {
                              _var_72 = 0ul;
                              _var_73 = *(generic64_t *)&_var_167;
                              _var_74 = *(generic64_t *)&_var_147;
                            _label_10:
                              *(generic64_t *)&_var_168 = _var_77 + 2ul + _var_72;
                              _var_74 = (_var_74 * 10ul & 4294967294ul) + _var_73;
                              _var_73 = (generic64_t)(int64_t)(int8_t)*(generic8_t *)*(generic64_t *)&_var_168 + 4294967248ul & 4294967295ul;
                              _var_72 = _var_72 + 1ul;
                              if ((generic8_t)((uint64_t)_var_73 > 9ul)) {
                                _var_70 = _var_74;
                                _var_71 = *(generic64_t *)&_var_168;
                              } else {
                                goto _label_10;
                              }
                            }
                          }
                        } else {
                        }
                      } else {
                      }
                    } else if (((generic8_t *)*(generic64_t *)&_var_153)[2ul] == 36u) {
                      _var_84 = ((generic64_t *)&_var_0)[3ul];
                      *(generic64_t *)&_var_157 = _var_84;
                      *(generic64_t *)&_var_158 = ((generic64_t *)&_var_0)[5ul];
                      ((generic32_t *)&_var_0)[12ul] = 1u;
                      _var_83 = *(generic64_t *)&_var_153 + 3ul;
                      ((generic32_t *)((*(generic64_t *)&_var_156 << 2ul) + *(generic64_t *)&_var_157))[4611686018427387856ul] = 10u;
                      _var_82 = ((generic64_t)(int64_t)(int8_t)((generic8_t *)*(generic64_t *)&_var_153)[1ul] << 4ul) + *(generic64_t *)&_var_158 + 18446744073709550848ul;
                    } else {
                    }
                  } else {
                    *(generic64_t *)&_var_154 = (generic64_t)(int64_t)(int8_t)*(generic8_t *)*(generic64_t *)&_var_153 + 4294967248ul & 4294967295ul;
                    _var_85 = *(generic64_t *)&_var_147;
                    _var_86 = *(generic64_t *)&_var_153;
                    if ((generic8_t)((uint64_t)*(generic64_t *)&_var_154 > 9ul)) {
                      _var_77 = _var_86;
                      _var_78 = (generic64_t)(uint64_t)(uint32_t)_var_93;
                      _var_76 = _var_85 & 4294967295ul;
                      _var_10 = 4294967295u;
                      _var_75 = *(generic64_t *)&_var_152;
                      if ((_var_85 & 2147483648ul) == 0ul) {
                      } else {
                      }
                    } else {
                      _var_87 = 0ul;
                      _var_88 = *(generic64_t *)&_var_154;
                      _var_89 = *(generic64_t *)&_var_147;
                    _label_11:
                      *(generic64_t *)&_var_155 = *(generic64_t *)&_var_153 + 1ul + _var_87;
                      _var_89 = (_var_89 * 10ul & 4294967294ul) + _var_88;
                      _var_88 = (generic64_t)(int64_t)(int8_t)*(generic8_t *)*(generic64_t *)&_var_155 + 4294967248ul & 4294967295ul;
                      _var_87 = _var_87 + 1ul;
                      if ((generic8_t)((uint64_t)_var_88 > 9ul)) {
                        _var_85 = _var_89;
                        _var_86 = *(generic64_t *)&_var_155;
                      } else {
                        goto _label_11;
                      }
                    }
                  }
                } else {
                  _var_98 = 0ul;
                  _var_99 = _var_93 + 4294967264u;
                  _var_100 = (generic32_t)(int32_t)(int8_t)_var_92;
                  _var_101 = _var_151;
                  _var_102 = _var_105;
                  _var_103 = 0ul;
                _label_12:
                  _var_97 = _var_100;
                  _var_96 = _var_101;
                  _var_95 = _var_102;
                  _var_94 = _var_103;
                  *(generic64_t *)&_var_122 = lshift((generic64_t)(uint64_t)(uint32_t)(_var_97 + 4294967233u), 4294967272u);
                  *(generic64_t *)&_var_123 = lshift((generic64_t)(uint64_t)(uint32_t)((_var_99 ^ 31u) & (_var_99 ^ _var_97 + 4294967233u)), 4294967276u);
                  if (((generic64_t)(75913ul >> (uint64_t)(uint8_t)(_var_96 & 31u)) & 1ul) == 0ul) {
                    _var_90 = _var_94;
                    _var_91 = _var_95;
                    _var_92 = _var_96;
                    _var_93 = _var_97;
                  } else {
                    _var_95 = _var_105 + 1ul + _var_98;
                    _var_102 = _var_102 + 1ul;
                    _var_103 = _var_103 | 1ul << (generic64_t)(uint64_t)(uint8_t)(_var_96 & 31u);
                    _var_94 = _var_103;
                    _var_101 = *(generic8_t *)_var_95;
                    _var_96 = _var_101;
                    _var_100 = (generic32_t)(int32_t)(int8_t)_var_96;
                    _var_97 = _var_100;
                    _var_99 = _var_97 + 4294967264u;
                    _var_98 = _var_98 + 1ul;
                    if ((generic8_t)((uint32_t)_var_99 > 31u)) {
                    } else {
                      goto _label_12;
                    }
                  }
                }
              } else {
              }
            }
          } else {
          }
        } else {
        }
      }
      case 0u: {
      }
      default: {
        goto _label_3;
      }
    }
  }
  __builtin_unreachable();
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
  ((generic32_t *)f)[35ul] = 0u;
  helper_lock();
  helper_unlock();
  if (((generic32_t *)f)[36ul] == 0u) {
  } else {
    helper_syscall_wrapper((void *)0ul, 2u, 4209230ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)f, (generic64_t)f + 140ul, 1ul, 129ul, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42, (void *)&_var_43, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47);
    if (_var_27 == 18446744073709551578ul) {
      helper_syscall_wrapper((void *)0ul, 2u, 4209244ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)f, (generic64_t)f + 140ul, 1ul, 1ul, _var_29, _var_30, _var_32, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, _var_34, _var_38, _var_41, _var_42, 274877906944ul, 127u, 2147549185ul, 0ul, _var_43, _var_45, _var_46, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
    } else {
    }
  }
  return;
}

_ABI(SystemV_x86_64)
int8_t *fmt_u(unreserved_uintmax_t x, int8_t *s) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic8_t _var_3[8];
  generic8_t _var_4[8];
  _var_0 = (generic64_t)s;
  if ((generic64_t)x == 0ul) {
  } else {
    _var_1 = 0ul;
    _var_2 = (generic64_t)x;
  _label_0:
    *(generic64_t *)&_var_3 = _var_1;
    *(generic64_t *)&_var_4 = _var_2;
    _var_2 = (generic64_t)((uint64_t)*(generic64_t *)&_var_4 / 10ul);
    *(generic8_t *)((generic64_t)s + 18446744073709551615ul - *(generic64_t *)&_var_3) = (generic8_t)(generic64_t)((uint64_t)*(generic64_t *)&_var_4 % 10ul) | 48u;
    _var_1 = *(generic64_t *)&_var_3 + 1ul;
    if ((generic8_t)((uint64_t)*(generic64_t *)&_var_4 < 10ul)) {
      _var_0 = (generic64_t)s + 18446744073709551615ul - *(generic64_t *)&_var_3;
    } else {
      goto _label_0;
    }
  }
  return (int8_t *)_var_0;
}

_ABI(SystemV_x86_64)
void out(FILE_ *f, const int8_t *s, size_t l) {
  struct _PACKED struct_567 {
    struct_724 *offset_0;
    uint8_t padding_at_8[32];
  };
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic8_t _var_4[4];
  generic8_t _var_5[8];
  generic8_t _var_6[8];
  generic8_t _var_7[8];
  generic8_t _var_8;
  generic8_t _var_9;
  if ((*(generic8_t *)f & 32u) == 0u) {
    if (((generic64_t *)f)[4ul] == 0ul) {
      *(int32_t *)&_var_4 = unreserved___towrite(f);
      if (*(generic32_t *)&_var_4 == 0u) {
        if ((generic8_t)((uint64_t)(((generic64_t *)f)[4ul] - ((generic64_t *)f)[5ul]) < (uint64_t)l)) {
        } else {
          _var_8 = (generic8_t)((int8_t)((generic8_t *)f)[139ul] < 0) ? 1u : (generic8_t)((generic64_t)l == 0ul);
          _var_0 = (generic64_t)s;
          _var_1 = (generic64_t)l;
          if (_var_8) {
            *(struct struct_718 **)&_var_7 = memcpy((struct struct_718 *)((generic64_t *)f)[5ul], (union union_596 *)_var_0, _var_1);
            ((generic64_t *)f)[5ul] = ((generic64_t *)f)[5ul] + _var_1;
          } else {
            _var_2 = 0ul;
            _var_3 = (generic64_t)l;
          _label_0:
            if (*(generic8_t *)((generic64_t)l + (generic64_t)s + (_var_2 ^ 18446744073709551615ul)) == 10u) {
              struct rawfunction_25 _var_10 = ((rawfunction_25 *)((generic64_t *)f)[9ul])((pointer_or_number64_t)/* undef */ (generic64_t){0}, (pointer_or_number64_t)_var_3, (pointer_or_number64_t)s, (pointer_or_number64_t)f, (pointer_or_number64_t)f, (pointer_or_number64_t)/* undef */ (generic64_t){0});
              *(pointer_or_number64_t *)&_var_5 = _var_10.;
              *(pointer_or_number64_t *)&_var_6 = _var_10.;
              if ((generic8_t)((uint64_t)_var_3 > (uint64_t)*(generic64_t *)&_var_5)) {
              } else {
                _var_1 = (generic64_t)l - _var_3;
                _var_0 = (generic64_t)l + (generic64_t)s - _var_2;
              }
            } else {
              _var_3 = _var_3 + 18446744073709551615ul;
              *(int8_t *)&_var_9 = (_var_2 ^ 18446744073709551615ul) == 0ul - (generic64_t)l;
              _var_2 = _var_2 + 1ul;
              if (_var_9) {
                _var_0 = (generic64_t)s;
                _var_1 = (generic64_t)l;
              } else {
                goto _label_0;
              }
            }
          }
        }
      } else {
      }
    } else {
    }
  } else {
  }
  return;
}

_ABI(SystemV_x86_64)
int32_t unreserved___towrite(FILE_ *f) {
  generic32_t _var_0;
  generic8_t _var_1[8];
  ((generic8_t *)f)[138ul] = ((generic8_t *)f)[138ul] + 255u | ((generic8_t *)f)[138ul];
  if ((*(generic32_t *)f & 8u) == 0u) {
    *(generic64_t *)&_var_1 = ((generic64_t *)f)[11ul];
    ((generic64_t *)f)[2ul] = 0ul;
    ((generic64_t *)f)[1ul] = 0ul;
    ((generic64_t *)f)[7ul] = *(generic64_t *)&_var_1;
    ((generic64_t *)f)[5ul] = *(generic64_t *)&_var_1;
    ((generic64_t *)f)[4ul] = *(generic64_t *)&_var_1 + ((generic64_t *)f)[12ul];
    _var_0 = 0u;
  } else {
    *(generic32_t *)f = *(generic32_t *)f | 32u;
    _var_0 = 4294967295u;
  }
  return (int32_t)_var_0;
}

_ABI(SystemV_x86_64)
struct_718 *memcpy(struct_718 *argument_0, union_596 *argument_1, generic64_t argument_2) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic64_t _var_6;
  generic64_t _var_7;
  generic64_t _var_8;
  generic64_t _var_9;
  generic64_t _var_10;
  generic64_t _var_11;
  generic64_t _var_12;
  generic64_t _var_13;
  generic64_t _var_14;
  generic8_t _var_15[8];
  generic8_t _var_16[8];
  generic8_t _var_17[8];
  generic8_t _var_18[8];
  generic8_t _var_19[8];
  generic8_t _var_20[8];
  generic8_t _var_21[8];
  generic8_t _var_22[8];
  generic8_t _var_23[8];
  generic8_t _var_24[8];
  generic8_t _var_25[8];
  generic8_t _var_26[8];
  generic8_t _var_27[8];
  _var_8 = (generic64_t)argument_0;
  _var_9 = argument_2;
  _var_10 = (generic64_t)argument_1;
  if ((generic8_t)((uint64_t)argument_2 < 8ul) ? 1u : (generic8_t)(((generic64_t)argument_0 & 7ul) == 0ul)) {
    _var_3 = _var_8;
    *(generic64_t *)&_var_20 = _var_9;
    _var_4 = _var_10;
    if ((generic8_t)((uint64_t)*(generic64_t *)&_var_20 < 8ul)) {
      if ((*(generic64_t *)&_var_20 & 7ul) == 0ul) {
      } else {
        _var_0 = _var_3;
        _var_1 = *(generic64_t *)&_var_20 & 7ul;
        _var_2 = _var_4;
      _label_0:
        *(generic64_t *)&_var_25 = _var_0;
        *(generic64_t *)&_var_26 = _var_1;
        *(generic64_t *)&_var_27 = _var_2;
        *(generic8_t *)*(generic64_t *)&_var_25 = *(generic8_t *)*(generic64_t *)&_var_27;
        _var_2 = *(generic64_t *)&_var_27 + 1ul;
        _var_0 = *(generic64_t *)&_var_25 + 1ul;
        _var_1 = *(generic64_t *)&_var_26 + 18446744073709551615ul & 4294967295ul;
        if (_var_1 == 0ul) {
        } else {
          goto _label_0;
        }
      }
    } else {
      *(generic64_t *)&_var_21 = _var_10 + (*(generic64_t *)&_var_20 & 18446744073709551608ul);
      *(generic64_t *)&_var_19 = _var_8 + (*(generic64_t *)&_var_20 & 18446744073709551608ul);
      _var_5 = 0ul;
      _var_6 = _var_10;
      _var_7 = _var_8;
    _label_1:
      *(generic64_t *)&_var_22 = _var_6;
      *(generic64_t *)&_var_23 = _var_7;
      _var_5 = _var_5 + 1ul;
      *(generic64_t *)&_var_24 = _var_5;
      *(generic64_t *)*(generic64_t *)&_var_23 = *(generic64_t *)*(generic64_t *)&_var_22;
      _var_6 = *(generic64_t *)&_var_22 + 8ul;
      _var_7 = *(generic64_t *)&_var_23 + 8ul;
      if ((generic64_t)((uint64_t)*(generic64_t *)&_var_20 >> 3ul) == *(generic64_t *)&_var_24) {
        _var_3 = *(generic64_t *)&_var_19;
        _var_4 = *(generic64_t *)&_var_21;
      } else {
        goto _label_1;
      }
    }
  } else {
    _var_11 = 0ul;
    _var_12 = (generic64_t)argument_0;
    _var_13 = argument_2;
    _var_14 = (generic64_t)argument_1;
  _label_2:
    *(generic64_t *)&_var_15 = _var_11;
    *(generic64_t *)&_var_16 = _var_12;
    *(generic64_t *)&_var_17 = _var_13;
    *(generic64_t *)&_var_18 = _var_14;
    *(generic8_t *)*(generic64_t *)&_var_16 = *(generic8_t *)*(generic64_t *)&_var_18;
    _var_14 = *(generic64_t *)&_var_18 + 1ul;
    _var_12 = *(generic64_t *)&_var_16 + 1ul;
    _var_13 = *(generic64_t *)&_var_17 + 18446744073709551615ul;
    _var_11 = *(generic64_t *)&_var_15 + 1ul;
    if (((generic64_t)argument_0 + 1ul + *(generic64_t *)&_var_15 & 7ul) == 0ul) {
      _var_8 = *(generic64_t *)&_var_16 + 1ul;
      _var_9 = *(generic64_t *)&_var_17 + 18446744073709551615ul;
      _var_10 = *(generic64_t *)&_var_18 + 1ul;
    } else {
      goto _label_2;
    }
  }
  return argument_0;
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
  generic64_t _var_69;
  generic64_t _var_70;
  generic64_t _var_71;
  generic64_t _var_72;
  generic64_t _var_73;
  generic64_t _var_74;
  generic64_t _var_75;
  generic64_t _var_76;
  generic64_t _var_77;
  generic8_t _var_78[8];
  generic8_t _var_79[4];
  generic8_t _var_80[4];
  generic8_t _var_81[4];
  generic8_t _var_82[4];
  generic8_t _var_83[4];
  generic8_t _var_84[4];
  generic8_t _var_85[4];
  generic8_t _var_86[4];
  if ((generic8_t)((uint32_t)type > 22u)) {
    return;
  } else if ((generic8_t)((uint32_t)((generic32_t)type + 4294967287u) > 11u)) {
  } else {
    switch (((generic64_t *)((generic64_t)(uint64_t)(uint32_t)((generic32_t)type + 4294967287u) << 3ul))[526852ul]) {
      case 4201944ul: {
        if ((generic8_t)((uint32_t)*(generic32_t *)ap > 47u)) {
          _var_70 = ((generic64_t *)ap)[1ul];
          ((generic64_t *)ap)[1ul] = _var_70 + 8ul;
        } else {
          *(generic32_t *)&_var_86 = *(generic32_t *)ap + 8u;
          _var_70 = ((generic64_t *)ap)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)ap;
          *(generic32_t *)ap = *(generic32_t *)&_var_86;
        }
        _var_69 = (generic64_t)(int64_t)(int32_t)*(generic32_t *)_var_70;
        *(generic64_t *)arg_ = _var_69;
      }
      case 4201983ul: {
        if ((generic8_t)((uint32_t)*(generic32_t *)ap > 47u)) {
          _var_71 = ((generic64_t *)ap)[1ul];
          ((generic64_t *)ap)[1ul] = _var_71 + 8ul;
        } else {
          *(generic32_t *)&_var_85 = *(generic32_t *)ap + 8u;
          _var_71 = ((generic64_t *)ap)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)ap;
          *(generic32_t *)ap = *(generic32_t *)&_var_85;
        }
        _var_69 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)_var_71;
      }
      case 4202019ul: {
        if ((generic8_t)((uint32_t)*(generic32_t *)ap > 47u)) {
          _var_72 = ((generic64_t *)ap)[1ul];
          ((generic64_t *)ap)[1ul] = _var_72 + 8ul;
        } else {
          *(generic32_t *)&_var_84 = *(generic32_t *)ap + 8u;
          _var_72 = ((generic64_t *)ap)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)ap;
          *(generic32_t *)ap = *(generic32_t *)&_var_84;
        }
        _var_69 = *(generic64_t *)_var_72;
      }
      case 4202056ul: {
        if ((generic8_t)((uint32_t)*(generic32_t *)ap > 47u)) {
          _var_73 = ((generic64_t *)ap)[1ul];
          ((generic64_t *)ap)[1ul] = _var_73 + 8ul;
        } else {
          *(generic32_t *)&_var_83 = *(generic32_t *)ap + 8u;
          _var_73 = ((generic64_t *)ap)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)ap;
          *(generic32_t *)ap = *(generic32_t *)&_var_83;
        }
        _var_69 = (generic64_t)(int64_t)(int16_t)*(generic16_t *)_var_73;
      }
      case 4202094ul: {
        if ((generic8_t)((uint32_t)*(generic32_t *)ap > 47u)) {
          _var_74 = ((generic64_t *)ap)[1ul];
          ((generic64_t *)ap)[1ul] = _var_74 + 8ul;
        } else {
          *(generic32_t *)&_var_82 = *(generic32_t *)ap + 8u;
          _var_74 = ((generic64_t *)ap)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)ap;
          *(generic32_t *)ap = *(generic32_t *)&_var_82;
        }
        _var_69 = (generic64_t)(uint64_t)(uint16_t)*(generic16_t *)_var_74;
      }
      case 4202134ul: {
        if ((generic8_t)((uint32_t)*(generic32_t *)ap > 47u)) {
          _var_75 = ((generic64_t *)ap)[1ul];
          ((generic64_t *)ap)[1ul] = _var_75 + 8ul;
        } else {
          *(generic32_t *)&_var_81 = *(generic32_t *)ap + 8u;
          _var_75 = ((generic64_t *)ap)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)ap;
          *(generic32_t *)ap = *(generic32_t *)&_var_81;
        }
        _var_69 = (generic64_t)(int64_t)(int8_t)*(generic8_t *)_var_75;
      }
      case 4202175ul: {
        if ((generic8_t)((uint32_t)*(generic32_t *)ap > 47u)) {
          _var_76 = ((generic64_t *)ap)[1ul];
          ((generic64_t *)ap)[1ul] = _var_76 + 8ul;
        } else {
          *(generic32_t *)&_var_80 = *(generic32_t *)ap + 8u;
          _var_76 = ((generic64_t *)ap)[2ul] + (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)ap;
          *(generic32_t *)ap = *(generic32_t *)&_var_80;
        }
        _var_69 = (generic64_t)(uint64_t)(uint8_t)*(generic8_t *)_var_76;
      }
      case 4202215ul: {
        if ((generic8_t)((uint32_t)((generic32_t *)ap)[1ul] > 175u)) {
          _var_77 = ((generic64_t *)ap)[1ul];
          ((generic64_t *)ap)[1ul] = _var_77 + 8ul;
        } else {
          *(generic32_t *)&_var_79 = ((generic32_t *)ap)[1ul] + 16u;
          _var_77 = ((generic64_t *)ap)[2ul] + (generic64_t)(uint64_t)(uint32_t)((generic32_t *)ap)[1ul];
          ((generic32_t *)ap)[1ul] = *(generic32_t *)&_var_79;
        }
        helper_fldl_ST0_wrapper((void *)0ul, *(generic64_t *)_var_77, 0u, 0u, 0u, 0u, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34);
        helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)arg_, _var_9, _var_18, _var_19, _var_20, _var_21, _var_22, _var_23, _var_24, _var_25, _var_26, _var_27, _var_28, _var_29, _var_30, _var_31, _var_32, _var_33);
        helper_fpop_wrapper((void *)0ul, _var_9, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8);
      }
      case 4202257ul: {
        *(generic64_t *)&_var_78 = ((generic64_t *)ap)[1ul] + 15ul & 18446744073709551600ul;
        ((generic64_t *)ap)[1ul] = *(generic64_t *)&_var_78 + 16ul;
        helper_fldt_ST0_wrapper((void *)0ul, *(generic64_t *)&_var_78, 0u, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47, (void *)&_var_48, (void *)&_var_49, (void *)&_var_50, (void *)&_var_51, (void *)&_var_52, (void *)&_var_53, (void *)&_var_54, (void *)&_var_55, (void *)&_var_56, (void *)&_var_57, (void *)&_var_58, (void *)&_var_59, (void *)&_var_60, (void *)&_var_61, (void *)&_var_62, (void *)&_var_63, (void *)&_var_64, (void *)&_var_65, (void *)&_var_66, (void *)&_var_67, (void *)&_var_68);
        helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)arg_, _var_44, _var_53, _var_54, _var_55, _var_56, _var_57, _var_58, _var_59, _var_60, _var_61, _var_62, _var_63, _var_64, _var_65, _var_66, _var_67, _var_68);
        helper_fpop_wrapper((void *)0ul, _var_44, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42, (void *)&_var_43);
      }
      default: {
      }
    }
  }
  __builtin_unreachable();
}

_ABI(SystemV_x86_64)
void pad(FILE_ *f, int8_t c, int32_t w, int32_t l, int32_t fl) {
  struct _PACKED struct_568 {
    struct_661 offset_0;
    uint8_t padding_at_63[217];
  };
  generic8_t _var_0[280];
  generic32_t _var_1;
  generic8_t _var_2[8];
  generic8_t _var_3;
  if (((generic32_t)fl & 73728u) == 0u) {
    if ((generic8_t)((int64_t)((generic64_t)(uint64_t)(uint32_t)l << 32ul) < (int64_t)((generic64_t)(uint64_t)(uint32_t)w << 32ul))) {
      *(struct struct_661 **)&_var_2 = memset((struct struct_661 *)&_var_0, (generic64_t)(int64_t)c & 4294967295ul, (generic8_t)((int32_t)(generic32_t)((generic64_t)(uint64_t)(uint32_t)w - (generic64_t)(uint64_t)(uint32_t)l) > 256) ? 256ul : (generic64_t)((int64_t)((generic64_t)(uint64_t)(uint32_t)w - (generic64_t)(uint64_t)(uint32_t)l << 32ul) >> 32l), (generic64_t)(uint64_t)(uint32_t)l, 0ul);
      if ((generic8_t)((int32_t)(generic32_t)((generic64_t)(uint64_t)(uint32_t)w - (generic64_t)(uint64_t)(uint32_t)l) > 255)) {
        _var_1 = 0u;
      _label_0:
        out(f, (const int8_t *)&_var_0, 256ul);
        _var_3 = (generic8_t)((int32_t)((generic32_t)w + 4294967040u - ((_var_1 << 8u) + (generic32_t)l)) > 255);
        _var_1 = _var_1 + 1u;
        if (_var_3) {
          goto _label_0;
        } else {
          out(f, (const int8_t *)&_var_0, (size_t)((int64_t)(((generic64_t)((uint64_t)((generic64_t)(uint64_t)(uint32_t)w - (generic64_t)(uint64_t)(uint32_t)l) >> 8ul) & 16777215ul) * 4294967040ul + ((generic64_t)(uint64_t)(uint32_t)w - (generic64_t)(uint64_t)(uint32_t)l) << 32ul) >> 32l));
        }
      } else {
      }
    } else {
    }
  } else {
  }
  return;
}

_ABI(SystemV_x86_64)
struct_661 *memset(struct_661 *argument_0, generic64_t argument_1, generic64_t argument_2, generic64_t argument_3, generic64_t argument_4) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic8_t _var_4[8];
  generic8_t _var_5[8];
  generic8_t _var_6[8];
  if ((generic8_t)((uint64_t)argument_2 > 126ul)) {
    ((generic64_t *)(argument_2 + (generic64_t)argument_0))[2305843009213693951ul] = (argument_1 & 255ul) * 72340172838076673ul;
    _var_2 = argument_2;
    _var_3 = (generic64_t)argument_0;
    if (((generic64_t)argument_0 & 15ul) == 0ul) {
    } else {
      *(generic64_t *)argument_0 = (argument_1 & 255ul) * 72340172838076673ul;
      ((generic64_t *)argument_0)[1ul] = (argument_1 & 255ul) * 72340172838076673ul;
      _var_2 = argument_2 - (0ul - (generic64_t)argument_0 & 15ul);
      _var_3 = (0ul - (generic64_t)argument_0 & 15ul) + (generic64_t)argument_0;
    }
    if ((generic8_t)((uint64_t)_var_2 < 8ul)) {
    } else {
      *(generic64_t *)&_var_4 = (generic64_t)((uint64_t)_var_2 >> 3ul);
      _var_0 = 0ul;
      _var_1 = _var_3;
    _label_0:
      *(generic64_t *)&_var_5 = _var_1;
      _var_0 = _var_0 + 1ul;
      *(generic64_t *)&_var_6 = _var_0;
      *(generic64_t *)*(generic64_t *)&_var_5 = (argument_1 & 255ul) * 72340172838076673ul;
      _var_1 = *(generic64_t *)&_var_5 + 8ul;
      if (*(generic64_t *)&_var_4 == *(generic64_t *)&_var_6) {
      } else {
        goto _label_0;
      }
    }
  } else if ((argument_2 & 4294967295ul) == 0ul) {
  } else {
    *(generic8_t *)argument_0 = (generic8_t)argument_1;
    ((generic8_t *)(argument_2 + (generic64_t)argument_0))[18446744073709551615ul] = (generic8_t)argument_1;
    if ((generic8_t)((uint64_t)(argument_2 & 4294967295ul) > 2ul)) {
      *(generic16_t *)((generic64_t)argument_0 + 1ul) = (generic16_t)((argument_1 & 255ul) * 72340172838076673ul);
      *(generic16_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551613ul) = (generic16_t)((argument_1 & 255ul) * 72340172838076673ul);
      if ((generic8_t)((uint64_t)(argument_2 & 4294967295ul) > 6ul)) {
        *(generic32_t *)((generic64_t)argument_0 + 3ul) = (generic32_t)((argument_1 & 255ul) * 72340172838076673ul);
        *(generic32_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551609ul) = (generic32_t)((argument_1 & 255ul) * 72340172838076673ul);
        if ((generic8_t)((uint64_t)(argument_2 & 4294967295ul) > 14ul)) {
          *(generic64_t *)((generic64_t)argument_0 + 7ul) = (argument_1 & 255ul) * 72340172838076673ul;
          *(generic64_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551601ul) = (argument_1 & 255ul) * 72340172838076673ul;
          if ((generic8_t)((uint64_t)(argument_2 & 4294967295ul) > 30ul)) {
            *(generic64_t *)((generic64_t)argument_0 + 15ul) = (argument_1 & 255ul) * 72340172838076673ul;
            *(generic64_t *)((generic64_t)argument_0 + 23ul) = (argument_1 & 255ul) * 72340172838076673ul;
            *(generic64_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551585ul) = (argument_1 & 255ul) * 72340172838076673ul;
            *(generic64_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551593ul) = (argument_1 & 255ul) * 72340172838076673ul;
            if ((generic8_t)((uint64_t)(argument_2 & 4294967295ul) > 62ul)) {
              *(generic64_t *)((generic64_t)argument_0 + 31ul) = (argument_1 & 255ul) * 72340172838076673ul;
              *(generic64_t *)((generic64_t)argument_0 + 39ul) = (argument_1 & 255ul) * 72340172838076673ul;
              *(generic64_t *)((generic64_t)argument_0 + 47ul) = (argument_1 & 255ul) * 72340172838076673ul;
              *(generic64_t *)((generic64_t)argument_0 + 55ul) = (argument_1 & 255ul) * 72340172838076673ul;
              *(generic64_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551553ul) = (argument_1 & 255ul) * 72340172838076673ul;
              *(generic64_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551561ul) = (argument_1 & 255ul) * 72340172838076673ul;
              *(generic64_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551569ul) = (argument_1 & 255ul) * 72340172838076673ul;
              *(generic64_t *)(argument_2 + (generic64_t)argument_0 + 18446744073709551577ul) = (argument_1 & 255ul) * 72340172838076673ul;
            } else {
            }
          } else {
          }
        } else {
        }
      } else {
      }
    } else {
    }
  }
  return argument_0;
}

_ABI(SystemV_x86_64)
int32_t fmt_fp(FILE_ *f, float128_t y, int32_t w, int32_t p, int32_t fl, int32_t t) {
  struct _PACKED struct_569 {
    generic64_t offset_0;
    generic64_t offset_8;
    generic32_t offset_16;
    generic8_t offset_20;
    uint8_t padding_at_21[3];
    union _PACKED union_674 {
      generic8_t member_0;
      generic32_t member_1;
      union_605 *member_2;
    } offset_24;
    union _PACKED union_675 {
      generic8_t *member_0;
      union_605 *member_1;
    } offset_32;
    uint8_t padding_at_40[8];
    union _PACKED union_676 {
      generic32_t member_0;
      union_605 *member_1;
    } offset_48;
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
    union _PACKED union_677 {
      union_605 member_0;
      struct _PACKED struct_678 {
        uint8_t padding_at_0[8];
        union_605 offset_8;
      } member_1;
      struct _PACKED struct_679 {
        uint8_t padding_at_0[9];
        union_605 offset_9;
      } member_2;
    } offset_107;
    uint8_t padding_at_148[7380];
  };
  generic64_t _var_0;
  generic16_t _var_1;
  generic64_t _var_2;
  generic16_t _var_3;
  generic64_t _var_4;
  generic16_t _var_5;
  generic64_t _var_6;
  generic16_t _var_7;
  generic64_t _var_8;
  generic16_t _var_9;
  generic64_t _var_10;
  generic16_t _var_11;
  generic64_t _var_12;
  generic16_t _var_13;
  generic64_t _var_14;
  generic16_t _var_15;
  generic64_t _var_16;
  generic16_t _var_17;
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
  generic64_t _var_34;
  generic16_t _var_35;
  generic64_t _var_36;
  generic16_t _var_37;
  generic64_t _var_38;
  generic16_t _var_39;
  generic64_t _var_40;
  generic16_t _var_41;
  generic64_t _var_42;
  generic16_t _var_43;
  generic64_t _var_44;
  generic16_t _var_45;
  generic64_t _var_46;
  generic16_t _var_47;
  generic64_t _var_48;
  generic16_t _var_49;
  generic64_t _var_50;
  generic16_t _var_51;
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
  generic32_t _var_64;
  generic8_t _var_65;
  generic8_t _var_66;
  generic8_t _var_67;
  generic8_t _var_68;
  generic8_t _var_69;
  generic8_t _var_70;
  generic8_t _var_71;
  generic8_t _var_72;
  generic64_t _var_73;
  generic16_t _var_74;
  generic64_t _var_75;
  generic16_t _var_76;
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
  generic32_t _var_89;
  generic8_t _var_90;
  generic8_t _var_91;
  generic8_t _var_92;
  generic8_t _var_93;
  generic8_t _var_94;
  generic8_t _var_95;
  generic8_t _var_96;
  generic8_t _var_97;
  generic64_t _var_98;
  generic16_t _var_99;
  generic64_t _var_100;
  generic16_t _var_101;
  generic64_t _var_102;
  generic16_t _var_103;
  generic64_t _var_104;
  generic16_t _var_105;
  generic64_t _var_106;
  generic16_t _var_107;
  generic64_t _var_108;
  generic16_t _var_109;
  generic64_t _var_110;
  generic16_t _var_111;
  generic64_t _var_112;
  generic16_t _var_113;
  generic32_t _var_114;
  generic8_t _var_115;
  generic8_t _var_116;
  generic8_t _var_117;
  generic8_t _var_118;
  generic8_t _var_119;
  generic8_t _var_120;
  generic8_t _var_121;
  generic8_t _var_122;
  generic64_t _var_123;
  generic8_t _var_124;
  generic64_t _var_125;
  generic16_t _var_126;
  generic64_t _var_127;
  generic16_t _var_128;
  generic64_t _var_129;
  generic16_t _var_130;
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
  generic8_t _var_143;
  generic64_t _var_144;
  generic16_t _var_145;
  generic64_t _var_146;
  generic16_t _var_147;
  generic64_t _var_148;
  generic16_t _var_149;
  generic64_t _var_150;
  generic16_t _var_151;
  generic64_t _var_152;
  generic16_t _var_153;
  generic64_t _var_154;
  generic16_t _var_155;
  generic64_t _var_156;
  generic16_t _var_157;
  generic64_t _var_158;
  generic16_t _var_159;
  generic64_t _var_160;
  generic16_t _var_161;
  generic32_t _var_162;
  generic8_t _var_163;
  generic8_t _var_164;
  generic8_t _var_165;
  generic8_t _var_166;
  generic8_t _var_167;
  generic8_t _var_168;
  generic8_t _var_169;
  generic8_t _var_170;
  generic32_t _var_171;
  generic8_t _var_172;
  generic8_t _var_173;
  generic8_t _var_174;
  generic8_t _var_175;
  generic8_t _var_176;
  generic8_t _var_177;
  generic8_t _var_178;
  generic8_t _var_179;
  generic64_t _var_180;
  generic16_t _var_181;
  generic64_t _var_182;
  generic16_t _var_183;
  generic64_t _var_184;
  generic16_t _var_185;
  generic64_t _var_186;
  generic16_t _var_187;
  generic64_t _var_188;
  generic16_t _var_189;
  generic64_t _var_190;
  generic16_t _var_191;
  generic64_t _var_192;
  generic16_t _var_193;
  generic64_t _var_194;
  generic16_t _var_195;
  generic8_t _var_196;
  generic32_t _var_197;
  generic8_t _var_198;
  generic8_t _var_199;
  generic8_t _var_200;
  generic8_t _var_201;
  generic8_t _var_202;
  generic8_t _var_203;
  generic8_t _var_204;
  generic8_t _var_205;
  generic64_t _var_206;
  generic16_t _var_207;
  generic64_t _var_208;
  generic16_t _var_209;
  generic64_t _var_210;
  generic16_t _var_211;
  generic64_t _var_212;
  generic16_t _var_213;
  generic64_t _var_214;
  generic16_t _var_215;
  generic64_t _var_216;
  generic16_t _var_217;
  generic64_t _var_218;
  generic16_t _var_219;
  generic64_t _var_220;
  generic16_t _var_221;
  generic8_t _var_222;
  generic32_t _var_223;
  generic8_t _var_224;
  generic8_t _var_225;
  generic8_t _var_226;
  generic8_t _var_227;
  generic8_t _var_228;
  generic8_t _var_229;
  generic8_t _var_230;
  generic8_t _var_231;
  generic64_t _var_232;
  generic16_t _var_233;
  generic64_t _var_234;
  generic16_t _var_235;
  generic64_t _var_236;
  generic16_t _var_237;
  generic64_t _var_238;
  generic16_t _var_239;
  generic64_t _var_240;
  generic16_t _var_241;
  generic64_t _var_242;
  generic16_t _var_243;
  generic64_t _var_244;
  generic16_t _var_245;
  generic64_t _var_246;
  generic16_t _var_247;
  generic8_t _var_248;
  generic32_t _var_249;
  generic8_t _var_250;
  generic8_t _var_251;
  generic8_t _var_252;
  generic8_t _var_253;
  generic8_t _var_254;
  generic8_t _var_255;
  generic8_t _var_256;
  generic8_t _var_257;
  generic64_t _var_258;
  generic16_t _var_259;
  generic64_t _var_260;
  generic16_t _var_261;
  generic64_t _var_262;
  generic16_t _var_263;
  generic64_t _var_264;
  generic16_t _var_265;
  generic64_t _var_266;
  generic16_t _var_267;
  generic64_t _var_268;
  generic16_t _var_269;
  generic64_t _var_270;
  generic16_t _var_271;
  generic64_t _var_272;
  generic16_t _var_273;
  generic32_t _var_274;
  generic8_t _var_275;
  generic8_t _var_276;
  generic8_t _var_277;
  generic8_t _var_278;
  generic8_t _var_279;
  generic8_t _var_280;
  generic8_t _var_281;
  generic8_t _var_282;
  generic64_t _var_283;
  generic16_t _var_284;
  generic64_t _var_285;
  generic16_t _var_286;
  generic64_t _var_287;
  generic16_t _var_288;
  generic64_t _var_289;
  generic16_t _var_290;
  generic64_t _var_291;
  generic16_t _var_292;
  generic64_t _var_293;
  generic16_t _var_294;
  generic64_t _var_295;
  generic16_t _var_296;
  generic64_t _var_297;
  generic16_t _var_298;
  generic32_t _var_299;
  generic8_t _var_300;
  generic8_t _var_301;
  generic8_t _var_302;
  generic8_t _var_303;
  generic8_t _var_304;
  generic8_t _var_305;
  generic8_t _var_306;
  generic8_t _var_307;
  generic64_t _var_308;
  generic16_t _var_309;
  generic64_t _var_310;
  generic16_t _var_311;
  generic64_t _var_312;
  generic16_t _var_313;
  generic64_t _var_314;
  generic16_t _var_315;
  generic64_t _var_316;
  generic16_t _var_317;
  generic64_t _var_318;
  generic16_t _var_319;
  generic64_t _var_320;
  generic16_t _var_321;
  generic64_t _var_322;
  generic16_t _var_323;
  generic32_t _var_324;
  generic8_t _var_325;
  generic8_t _var_326;
  generic8_t _var_327;
  generic8_t _var_328;
  generic8_t _var_329;
  generic8_t _var_330;
  generic8_t _var_331;
  generic8_t _var_332;
  generic64_t _var_333;
  generic16_t _var_334;
  generic64_t _var_335;
  generic16_t _var_336;
  generic64_t _var_337;
  generic16_t _var_338;
  generic64_t _var_339;
  generic16_t _var_340;
  generic64_t _var_341;
  generic16_t _var_342;
  generic64_t _var_343;
  generic16_t _var_344;
  generic64_t _var_345;
  generic16_t _var_346;
  generic64_t _var_347;
  generic16_t _var_348;
  generic32_t _var_349;
  generic8_t _var_350;
  generic8_t _var_351;
  generic8_t _var_352;
  generic8_t _var_353;
  generic8_t _var_354;
  generic8_t _var_355;
  generic8_t _var_356;
  generic8_t _var_357;
  generic64_t _var_358;
  generic16_t _var_359;
  generic64_t _var_360;
  generic16_t _var_361;
  generic64_t _var_362;
  generic16_t _var_363;
  generic64_t _var_364;
  generic16_t _var_365;
  generic64_t _var_366;
  generic16_t _var_367;
  generic64_t _var_368;
  generic16_t _var_369;
  generic64_t _var_370;
  generic16_t _var_371;
  generic64_t _var_372;
  generic16_t _var_373;
  generic32_t _var_374;
  generic8_t _var_375;
  generic8_t _var_376;
  generic8_t _var_377;
  generic8_t _var_378;
  generic8_t _var_379;
  generic8_t _var_380;
  generic8_t _var_381;
  generic8_t _var_382;
  generic64_t _var_383;
  generic16_t _var_384;
  generic64_t _var_385;
  generic16_t _var_386;
  generic64_t _var_387;
  generic16_t _var_388;
  generic64_t _var_389;
  generic16_t _var_390;
  generic64_t _var_391;
  generic16_t _var_392;
  generic64_t _var_393;
  generic16_t _var_394;
  generic64_t _var_395;
  generic16_t _var_396;
  generic64_t _var_397;
  generic16_t _var_398;
  generic32_t _var_399;
  generic8_t _var_400;
  generic8_t _var_401;
  generic8_t _var_402;
  generic8_t _var_403;
  generic8_t _var_404;
  generic8_t _var_405;
  generic8_t _var_406;
  generic8_t _var_407;
  generic64_t _var_408;
  generic16_t _var_409;
  generic64_t _var_410;
  generic16_t _var_411;
  generic64_t _var_412;
  generic16_t _var_413;
  generic64_t _var_414;
  generic16_t _var_415;
  generic64_t _var_416;
  generic16_t _var_417;
  generic64_t _var_418;
  generic16_t _var_419;
  generic64_t _var_420;
  generic16_t _var_421;
  generic64_t _var_422;
  generic16_t _var_423;
  generic32_t _var_424;
  generic8_t _var_425;
  generic8_t _var_426;
  generic8_t _var_427;
  generic8_t _var_428;
  generic8_t _var_429;
  generic8_t _var_430;
  generic8_t _var_431;
  generic8_t _var_432;
  generic64_t _var_433;
  generic16_t _var_434;
  generic64_t _var_435;
  generic16_t _var_436;
  generic64_t _var_437;
  generic16_t _var_438;
  generic64_t _var_439;
  generic16_t _var_440;
  generic64_t _var_441;
  generic16_t _var_442;
  generic64_t _var_443;
  generic16_t _var_444;
  generic64_t _var_445;
  generic16_t _var_446;
  generic64_t _var_447;
  generic16_t _var_448;
  generic32_t _var_449;
  generic8_t _var_450;
  generic8_t _var_451;
  generic8_t _var_452;
  generic8_t _var_453;
  generic8_t _var_454;
  generic8_t _var_455;
  generic8_t _var_456;
  generic8_t _var_457;
  generic64_t _var_458;
  generic16_t _var_459;
  generic64_t _var_460;
  generic16_t _var_461;
  generic64_t _var_462;
  generic16_t _var_463;
  generic64_t _var_464;
  generic16_t _var_465;
  generic64_t _var_466;
  generic16_t _var_467;
  generic64_t _var_468;
  generic16_t _var_469;
  generic64_t _var_470;
  generic16_t _var_471;
  generic64_t _var_472;
  generic16_t _var_473;
  generic64_t _var_474;
  generic8_t _var_475;
  generic64_t _var_476;
  generic16_t _var_477;
  generic64_t _var_478;
  generic8_t _var_479;
  generic64_t _var_480;
  generic16_t _var_481;
  generic32_t _var_482;
  generic8_t _var_483;
  generic8_t _var_484;
  generic8_t _var_485;
  generic8_t _var_486;
  generic8_t _var_487;
  generic8_t _var_488;
  generic8_t _var_489;
  generic8_t _var_490;
  generic64_t _var_491;
  generic16_t _var_492;
  generic64_t _var_493;
  generic16_t _var_494;
  generic64_t _var_495;
  generic16_t _var_496;
  generic64_t _var_497;
  generic16_t _var_498;
  generic64_t _var_499;
  generic16_t _var_500;
  generic64_t _var_501;
  generic16_t _var_502;
  generic64_t _var_503;
  generic16_t _var_504;
  generic64_t _var_505;
  generic16_t _var_506;
  generic8_t _var_507;
  generic64_t _var_508;
  generic16_t _var_509;
  generic64_t _var_510;
  generic16_t _var_511;
  generic64_t _var_512;
  generic16_t _var_513;
  generic64_t _var_514;
  generic16_t _var_515;
  generic64_t _var_516;
  generic16_t _var_517;
  generic64_t _var_518;
  generic16_t _var_519;
  generic64_t _var_520;
  generic16_t _var_521;
  generic64_t _var_522;
  generic16_t _var_523;
  generic8_t _var_524;
  generic64_t _var_525;
  generic16_t _var_526;
  generic64_t _var_527;
  generic16_t _var_528;
  generic64_t _var_529;
  generic16_t _var_530;
  generic64_t _var_531;
  generic16_t _var_532;
  generic64_t _var_533;
  generic16_t _var_534;
  generic64_t _var_535;
  generic16_t _var_536;
  generic64_t _var_537;
  generic16_t _var_538;
  generic64_t _var_539;
  generic16_t _var_540;
  generic32_t _var_541;
  generic8_t _var_542;
  generic8_t _var_543;
  generic8_t _var_544;
  generic8_t _var_545;
  generic8_t _var_546;
  generic8_t _var_547;
  generic8_t _var_548;
  generic8_t _var_549;
  generic64_t _var_550;
  generic16_t _var_551;
  generic64_t _var_552;
  generic16_t _var_553;
  generic64_t _var_554;
  generic16_t _var_555;
  generic64_t _var_556;
  generic16_t _var_557;
  generic64_t _var_558;
  generic16_t _var_559;
  generic64_t _var_560;
  generic16_t _var_561;
  generic64_t _var_562;
  generic16_t _var_563;
  generic64_t _var_564;
  generic16_t _var_565;
  generic8_t _var_566;
  generic64_t _var_567;
  generic16_t _var_568;
  generic64_t _var_569;
  generic16_t _var_570;
  generic64_t _var_571;
  generic16_t _var_572;
  generic64_t _var_573;
  generic16_t _var_574;
  generic64_t _var_575;
  generic16_t _var_576;
  generic64_t _var_577;
  generic16_t _var_578;
  generic64_t _var_579;
  generic16_t _var_580;
  generic64_t _var_581;
  generic16_t _var_582;
  generic8_t _var_583;
  generic64_t _var_584;
  generic16_t _var_585;
  generic64_t _var_586;
  generic16_t _var_587;
  generic64_t _var_588;
  generic16_t _var_589;
  generic64_t _var_590;
  generic16_t _var_591;
  generic64_t _var_592;
  generic16_t _var_593;
  generic64_t _var_594;
  generic16_t _var_595;
  generic64_t _var_596;
  generic16_t _var_597;
  generic64_t _var_598;
  generic16_t _var_599;
  generic64_t _var_600;
  generic16_t _var_601;
  generic64_t _var_602;
  generic16_t _var_603;
  generic64_t _var_604;
  generic16_t _var_605;
  generic64_t _var_606;
  generic16_t _var_607;
  generic64_t _var_608;
  generic16_t _var_609;
  generic64_t _var_610;
  generic16_t _var_611;
  generic64_t _var_612;
  generic16_t _var_613;
  generic64_t _var_614;
  generic16_t _var_615;
  generic64_t _var_616;
  generic16_t _var_617;
  generic64_t _var_618;
  generic16_t _var_619;
  generic64_t _var_620;
  generic16_t _var_621;
  generic64_t _var_622;
  generic16_t _var_623;
  generic64_t _var_624;
  generic16_t _var_625;
  generic64_t _var_626;
  generic16_t _var_627;
  generic64_t _var_628;
  generic16_t _var_629;
  generic64_t _var_630;
  generic16_t _var_631;
  generic64_t _var_632;
  generic16_t _var_633;
  generic8_t _var_634;
  generic64_t _var_635;
  generic16_t _var_636;
  generic64_t _var_637;
  generic16_t _var_638;
  generic64_t _var_639;
  generic16_t _var_640;
  generic64_t _var_641;
  generic16_t _var_642;
  generic64_t _var_643;
  generic16_t _var_644;
  generic64_t _var_645;
  generic16_t _var_646;
  generic64_t _var_647;
  generic16_t _var_648;
  generic64_t _var_649;
  generic16_t _var_650;
  generic64_t _var_651;
  generic16_t _var_652;
  generic8_t _var_653;
  generic64_t _var_654;
  generic16_t _var_655;
  generic16_t _var_656;
  generic8_t _var_657;
  generic8_t _var_658;
  generic32_t _var_659;
  generic8_t _var_660;
  generic8_t _var_661;
  generic8_t _var_662;
  generic8_t _var_663;
  generic8_t _var_664;
  generic8_t _var_665;
  generic8_t _var_666;
  generic8_t _var_667;
  generic8_t _var_668;
  generic16_t _var_669;
  generic8_t _var_670;
  generic8_t _var_671;
  generic64_t _var_672;
  generic16_t _var_673;
  generic64_t _var_674;
  generic16_t _var_675;
  generic64_t _var_676;
  generic16_t _var_677;
  generic64_t _var_678;
  generic16_t _var_679;
  generic64_t _var_680;
  generic16_t _var_681;
  generic64_t _var_682;
  generic16_t _var_683;
  generic64_t _var_684;
  generic16_t _var_685;
  generic64_t _var_686;
  generic16_t _var_687;
  generic32_t _var_688;
  generic8_t _var_689;
  generic8_t _var_690;
  generic8_t _var_691;
  generic8_t _var_692;
  generic8_t _var_693;
  generic8_t _var_694;
  generic8_t _var_695;
  generic8_t _var_696;
  generic64_t _var_697;
  generic8_t _var_698;
  generic64_t _var_699;
  generic16_t _var_700;
  generic64_t _var_701;
  generic16_t _var_702;
  generic64_t _var_703;
  generic16_t _var_704;
  generic64_t _var_705;
  generic16_t _var_706;
  generic64_t _var_707;
  generic16_t _var_708;
  generic64_t _var_709;
  generic16_t _var_710;
  generic64_t _var_711;
  generic16_t _var_712;
  generic64_t _var_713;
  generic16_t _var_714;
  generic64_t _var_715;
  generic16_t _var_716;
  generic8_t _var_717;
  generic8_t _var_718;
  generic64_t _var_719;
  generic16_t _var_720;
  generic32_t _var_721;
  generic8_t _var_722;
  generic8_t _var_723;
  generic8_t _var_724;
  generic8_t _var_725;
  generic8_t _var_726;
  generic8_t _var_727;
  generic8_t _var_728;
  generic8_t _var_729;
  generic64_t _var_730;
  generic16_t _var_731;
  generic64_t _var_732;
  generic16_t _var_733;
  generic64_t _var_734;
  generic16_t _var_735;
  generic64_t _var_736;
  generic16_t _var_737;
  generic64_t _var_738;
  generic16_t _var_739;
  generic64_t _var_740;
  generic16_t _var_741;
  generic64_t _var_742;
  generic16_t _var_743;
  generic64_t _var_744;
  generic16_t _var_745;
  generic8_t _var_746;
  generic32_t _var_747;
  generic8_t _var_748;
  generic8_t _var_749;
  generic8_t _var_750;
  generic8_t _var_751;
  generic8_t _var_752;
  generic8_t _var_753;
  generic8_t _var_754;
  generic8_t _var_755;
  generic64_t _var_756;
  generic16_t _var_757;
  generic64_t _var_758;
  generic16_t _var_759;
  generic64_t _var_760;
  generic16_t _var_761;
  generic64_t _var_762;
  generic16_t _var_763;
  generic64_t _var_764;
  generic16_t _var_765;
  generic64_t _var_766;
  generic16_t _var_767;
  generic64_t _var_768;
  generic16_t _var_769;
  generic64_t _var_770;
  generic16_t _var_771;
  generic16_t _var_772;
  generic8_t _var_773;
  generic8_t _var_774;
  generic32_t _var_775;
  generic8_t _var_776;
  generic8_t _var_777;
  generic8_t _var_778;
  generic8_t _var_779;
  generic8_t _var_780;
  generic8_t _var_781;
  generic8_t _var_782;
  generic8_t _var_783;
  generic8_t _var_784;
  generic16_t _var_785;
  generic8_t _var_786;
  generic8_t _var_787;
  generic64_t _var_788;
  generic16_t _var_789;
  generic64_t _var_790;
  generic16_t _var_791;
  generic64_t _var_792;
  generic16_t _var_793;
  generic64_t _var_794;
  generic16_t _var_795;
  generic64_t _var_796;
  generic16_t _var_797;
  generic64_t _var_798;
  generic16_t _var_799;
  generic64_t _var_800;
  generic16_t _var_801;
  generic64_t _var_802;
  generic16_t _var_803;
  generic32_t _var_804;
  generic8_t _var_805;
  generic8_t _var_806;
  generic8_t _var_807;
  generic8_t _var_808;
  generic8_t _var_809;
  generic8_t _var_810;
  generic8_t _var_811;
  generic8_t _var_812;
  generic64_t _var_813;
  generic16_t _var_814;
  generic64_t _var_815;
  generic16_t _var_816;
  generic64_t _var_817;
  generic16_t _var_818;
  generic64_t _var_819;
  generic16_t _var_820;
  generic64_t _var_821;
  generic16_t _var_822;
  generic64_t _var_823;
  generic16_t _var_824;
  generic64_t _var_825;
  generic16_t _var_826;
  generic64_t _var_827;
  generic16_t _var_828;
  generic64_t _var_829;
  generic16_t _var_830;
  generic64_t _var_831;
  generic16_t _var_832;
  generic64_t _var_833;
  generic16_t _var_834;
  generic64_t _var_835;
  generic16_t _var_836;
  generic64_t _var_837;
  generic16_t _var_838;
  generic64_t _var_839;
  generic16_t _var_840;
  generic64_t _var_841;
  generic16_t _var_842;
  generic64_t _var_843;
  generic16_t _var_844;
  generic8_t _var_845;
  generic64_t _var_846;
  generic16_t _var_847;
  generic32_t _var_848;
  generic8_t _var_849;
  generic8_t _var_850;
  generic8_t _var_851;
  generic8_t _var_852;
  generic8_t _var_853;
  generic8_t _var_854;
  generic8_t _var_855;
  generic8_t _var_856;
  generic64_t _var_857;
  generic16_t _var_858;
  generic64_t _var_859;
  generic16_t _var_860;
  generic64_t _var_861;
  generic16_t _var_862;
  generic64_t _var_863;
  generic16_t _var_864;
  generic64_t _var_865;
  generic16_t _var_866;
  generic64_t _var_867;
  generic16_t _var_868;
  generic64_t _var_869;
  generic16_t _var_870;
  generic64_t _var_871;
  generic16_t _var_872;
  generic32_t _var_873;
  generic8_t _var_874;
  generic8_t _var_875;
  generic8_t _var_876;
  generic8_t _var_877;
  generic8_t _var_878;
  generic8_t _var_879;
  generic8_t _var_880;
  generic8_t _var_881;
  generic64_t _var_882;
  generic16_t _var_883;
  generic64_t _var_884;
  generic16_t _var_885;
  generic64_t _var_886;
  generic16_t _var_887;
  generic64_t _var_888;
  generic16_t _var_889;
  generic64_t _var_890;
  generic16_t _var_891;
  generic64_t _var_892;
  generic16_t _var_893;
  generic64_t _var_894;
  generic16_t _var_895;
  generic64_t _var_896;
  generic16_t _var_897;
  generic64_t _var_898;
  generic16_t _var_899;
  generic64_t _var_900;
  generic16_t _var_901;
  generic64_t _var_902;
  generic16_t _var_903;
  generic64_t _var_904;
  generic16_t _var_905;
  generic64_t _var_906;
  generic16_t _var_907;
  generic64_t _var_908;
  generic16_t _var_909;
  generic64_t _var_910;
  generic16_t _var_911;
  generic64_t _var_912;
  generic16_t _var_913;
  generic32_t _var_914;
  generic8_t _var_915;
  generic8_t _var_916;
  generic8_t _var_917;
  generic8_t _var_918;
  generic8_t _var_919;
  generic8_t _var_920;
  generic8_t _var_921;
  generic8_t _var_922;
  generic32_t _var_923;
  generic8_t _var_924;
  generic8_t _var_925;
  generic8_t _var_926;
  generic8_t _var_927;
  generic8_t _var_928;
  generic8_t _var_929;
  generic8_t _var_930;
  generic8_t _var_931;
  generic64_t _var_932;
  generic16_t _var_933;
  generic64_t _var_934;
  generic16_t _var_935;
  generic64_t _var_936;
  generic16_t _var_937;
  generic64_t _var_938;
  generic16_t _var_939;
  generic64_t _var_940;
  generic16_t _var_941;
  generic64_t _var_942;
  generic16_t _var_943;
  generic64_t _var_944;
  generic16_t _var_945;
  generic64_t _var_946;
  generic16_t _var_947;
  generic32_t _var_948;
  generic8_t _var_949;
  generic8_t _var_950;
  generic8_t _var_951;
  generic8_t _var_952;
  generic8_t _var_953;
  generic8_t _var_954;
  generic8_t _var_955;
  generic8_t _var_956;
  generic64_t _var_957;
  generic16_t _var_958;
  generic64_t _var_959;
  generic16_t _var_960;
  generic64_t _var_961;
  generic16_t _var_962;
  generic64_t _var_963;
  generic16_t _var_964;
  generic64_t _var_965;
  generic16_t _var_966;
  generic64_t _var_967;
  generic16_t _var_968;
  generic64_t _var_969;
  generic16_t _var_970;
  generic64_t _var_971;
  generic16_t _var_972;
  generic8_t _var_973;
  generic8_t _var_974;
  generic64_t _var_975;
  generic16_t _var_976;
  generic32_t _var_977;
  generic8_t _var_978;
  generic8_t _var_979;
  generic8_t _var_980;
  generic8_t _var_981;
  generic8_t _var_982;
  generic8_t _var_983;
  generic8_t _var_984;
  generic8_t _var_985;
  generic64_t _var_986;
  generic16_t _var_987;
  generic64_t _var_988;
  generic16_t _var_989;
  generic64_t _var_990;
  generic16_t _var_991;
  generic64_t _var_992;
  generic16_t _var_993;
  generic64_t _var_994;
  generic16_t _var_995;
  generic64_t _var_996;
  generic16_t _var_997;
  generic64_t _var_998;
  generic16_t _var_999;
  generic64_t _var_1000;
  generic16_t _var_1001;
  generic8_t _var_1002;
  generic32_t _var_1003;
  generic8_t _var_1004;
  generic8_t _var_1005;
  generic8_t _var_1006;
  generic8_t _var_1007;
  generic8_t _var_1008;
  generic8_t _var_1009;
  generic8_t _var_1010;
  generic8_t _var_1011;
  generic64_t _var_1012;
  generic16_t _var_1013;
  generic64_t _var_1014;
  generic16_t _var_1015;
  generic64_t _var_1016;
  generic16_t _var_1017;
  generic64_t _var_1018;
  generic16_t _var_1019;
  generic64_t _var_1020;
  generic16_t _var_1021;
  generic64_t _var_1022;
  generic16_t _var_1023;
  generic64_t _var_1024;
  generic16_t _var_1025;
  generic64_t _var_1026;
  generic16_t _var_1027;
  generic64_t _var_1028;
  generic8_t _var_1029;
  generic64_t _var_1030;
  generic16_t _var_1031;
  generic32_t _var_1032;
  generic8_t _var_1033;
  generic8_t _var_1034;
  generic8_t _var_1035;
  generic8_t _var_1036;
  generic8_t _var_1037;
  generic8_t _var_1038;
  generic8_t _var_1039;
  generic8_t _var_1040;
  generic64_t _var_1041;
  generic16_t _var_1042;
  generic64_t _var_1043;
  generic16_t _var_1044;
  generic64_t _var_1045;
  generic16_t _var_1046;
  generic64_t _var_1047;
  generic16_t _var_1048;
  generic64_t _var_1049;
  generic16_t _var_1050;
  generic64_t _var_1051;
  generic16_t _var_1052;
  generic64_t _var_1053;
  generic16_t _var_1054;
  generic64_t _var_1055;
  generic16_t _var_1056;
  generic8_t _var_1057;
  generic32_t _var_1058;
  generic8_t _var_1059;
  generic8_t _var_1060;
  generic8_t _var_1061;
  generic8_t _var_1062;
  generic8_t _var_1063;
  generic8_t _var_1064;
  generic8_t _var_1065;
  generic8_t _var_1066;
  generic64_t _var_1067;
  generic16_t _var_1068;
  generic64_t _var_1069;
  generic16_t _var_1070;
  generic64_t _var_1071;
  generic16_t _var_1072;
  generic64_t _var_1073;
  generic16_t _var_1074;
  generic64_t _var_1075;
  generic16_t _var_1076;
  generic64_t _var_1077;
  generic16_t _var_1078;
  generic64_t _var_1079;
  generic16_t _var_1080;
  generic64_t _var_1081;
  generic16_t _var_1082;
  generic64_t _var_1083;
  generic8_t _var_1084;
  generic64_t _var_1085;
  generic16_t _var_1086;
  generic64_t _var_1087;
  generic16_t _var_1088;
  generic64_t _var_1089;
  generic16_t _var_1090;
  generic64_t _var_1091;
  generic16_t _var_1092;
  generic64_t _var_1093;
  generic16_t _var_1094;
  generic64_t _var_1095;
  generic16_t _var_1096;
  generic64_t _var_1097;
  generic16_t _var_1098;
  generic64_t _var_1099;
  generic16_t _var_1100;
  generic64_t _var_1101;
  generic16_t _var_1102;
  generic64_t _var_1103;
  generic16_t _var_1104;
  generic64_t _var_1105;
  generic16_t _var_1106;
  generic64_t _var_1107;
  generic16_t _var_1108;
  generic64_t _var_1109;
  generic16_t _var_1110;
  generic64_t _var_1111;
  generic16_t _var_1112;
  generic64_t _var_1113;
  generic16_t _var_1114;
  generic64_t _var_1115;
  generic16_t _var_1116;
  generic64_t _var_1117;
  generic16_t _var_1118;
  generic32_t _var_1119;
  generic8_t _var_1120;
  generic8_t _var_1121;
  generic8_t _var_1122;
  generic8_t _var_1123;
  generic8_t _var_1124;
  generic8_t _var_1125;
  generic8_t _var_1126;
  generic8_t _var_1127;
  generic64_t _var_1128;
  generic16_t _var_1129;
  generic64_t _var_1130;
  generic16_t _var_1131;
  generic64_t _var_1132;
  generic16_t _var_1133;
  generic64_t _var_1134;
  generic16_t _var_1135;
  generic64_t _var_1136;
  generic16_t _var_1137;
  generic64_t _var_1138;
  generic16_t _var_1139;
  generic64_t _var_1140;
  generic16_t _var_1141;
  generic64_t _var_1142;
  generic16_t _var_1143;
  generic8_t _var_1144;
  generic64_t _var_1145;
  generic16_t _var_1146;
  generic32_t _var_1147;
  generic8_t _var_1148;
  generic8_t _var_1149;
  generic8_t _var_1150;
  generic8_t _var_1151;
  generic8_t _var_1152;
  generic8_t _var_1153;
  generic8_t _var_1154;
  generic8_t _var_1155;
  generic64_t _var_1156;
  generic8_t _var_1157;
  generic64_t _var_1158;
  generic16_t _var_1159;
  generic32_t _var_1160;
  generic8_t _var_1161;
  generic8_t _var_1162;
  generic8_t _var_1163;
  generic8_t _var_1164;
  generic8_t _var_1165;
  generic8_t _var_1166;
  generic8_t _var_1167;
  generic8_t _var_1168;
  generic32_t _var_1169;
  generic8_t _var_1170;
  generic8_t _var_1171;
  generic8_t _var_1172;
  generic8_t _var_1173;
  generic8_t _var_1174;
  generic8_t _var_1175;
  generic8_t _var_1176;
  generic8_t _var_1177;
  generic64_t _var_1178;
  generic16_t _var_1179;
  generic64_t _var_1180;
  generic16_t _var_1181;
  generic64_t _var_1182;
  generic16_t _var_1183;
  generic64_t _var_1184;
  generic16_t _var_1185;
  generic64_t _var_1186;
  generic16_t _var_1187;
  generic64_t _var_1188;
  generic16_t _var_1189;
  generic64_t _var_1190;
  generic16_t _var_1191;
  generic64_t _var_1192;
  generic16_t _var_1193;
  generic32_t _var_1194;
  generic8_t _var_1195;
  generic8_t _var_1196;
  generic8_t _var_1197;
  generic8_t _var_1198;
  generic8_t _var_1199;
  generic8_t _var_1200;
  generic8_t _var_1201;
  generic8_t _var_1202;
  generic32_t _var_1203;
  generic8_t _var_1204;
  generic8_t _var_1205;
  generic8_t _var_1206;
  generic8_t _var_1207;
  generic8_t _var_1208;
  generic8_t _var_1209;
  generic8_t _var_1210;
  generic8_t _var_1211;
  generic64_t _var_1212;
  generic16_t _var_1213;
  generic64_t _var_1214;
  generic16_t _var_1215;
  generic64_t _var_1216;
  generic16_t _var_1217;
  generic64_t _var_1218;
  generic16_t _var_1219;
  generic64_t _var_1220;
  generic16_t _var_1221;
  generic64_t _var_1222;
  generic16_t _var_1223;
  generic64_t _var_1224;
  generic16_t _var_1225;
  generic64_t _var_1226;
  generic16_t _var_1227;
  generic32_t _var_1228;
  generic8_t _var_1229;
  generic8_t _var_1230;
  generic8_t _var_1231;
  generic8_t _var_1232;
  generic8_t _var_1233;
  generic8_t _var_1234;
  generic8_t _var_1235;
  generic8_t _var_1236;
  generic64_t _var_1237;
  generic16_t _var_1238;
  generic64_t _var_1239;
  generic16_t _var_1240;
  generic64_t _var_1241;
  generic16_t _var_1242;
  generic64_t _var_1243;
  generic16_t _var_1244;
  generic64_t _var_1245;
  generic16_t _var_1246;
  generic64_t _var_1247;
  generic16_t _var_1248;
  generic64_t _var_1249;
  generic16_t _var_1250;
  generic64_t _var_1251;
  generic16_t _var_1252;
  generic32_t _var_1253;
  generic8_t _var_1254;
  generic8_t _var_1255;
  generic8_t _var_1256;
  generic8_t _var_1257;
  generic8_t _var_1258;
  generic8_t _var_1259;
  generic8_t _var_1260;
  generic8_t _var_1261;
  generic64_t _var_1262;
  generic16_t _var_1263;
  generic64_t _var_1264;
  generic16_t _var_1265;
  generic64_t _var_1266;
  generic16_t _var_1267;
  generic64_t _var_1268;
  generic16_t _var_1269;
  generic64_t _var_1270;
  generic16_t _var_1271;
  generic64_t _var_1272;
  generic16_t _var_1273;
  generic64_t _var_1274;
  generic16_t _var_1275;
  generic64_t _var_1276;
  generic16_t _var_1277;
  generic32_t _var_1278;
  generic8_t _var_1279;
  generic8_t _var_1280;
  generic8_t _var_1281;
  generic8_t _var_1282;
  generic8_t _var_1283;
  generic8_t _var_1284;
  generic8_t _var_1285;
  generic8_t _var_1286;
  generic32_t _var_1287;
  generic8_t _var_1288;
  generic8_t _var_1289;
  generic8_t _var_1290;
  generic8_t _var_1291;
  generic8_t _var_1292;
  generic8_t _var_1293;
  generic8_t _var_1294;
  generic8_t _var_1295;
  generic64_t _var_1296;
  generic16_t _var_1297;
  generic64_t _var_1298;
  generic16_t _var_1299;
  generic64_t _var_1300;
  generic16_t _var_1301;
  generic64_t _var_1302;
  generic16_t _var_1303;
  generic64_t _var_1304;
  generic16_t _var_1305;
  generic64_t _var_1306;
  generic16_t _var_1307;
  generic64_t _var_1308;
  generic16_t _var_1309;
  generic64_t _var_1310;
  generic16_t _var_1311;
  generic32_t _var_1312;
  generic8_t _var_1313;
  generic8_t _var_1314;
  generic8_t _var_1315;
  generic8_t _var_1316;
  generic8_t _var_1317;
  generic8_t _var_1318;
  generic8_t _var_1319;
  generic8_t _var_1320;
  generic32_t _var_1321;
  generic8_t _var_1322;
  generic8_t _var_1323;
  generic8_t _var_1324;
  generic8_t _var_1325;
  generic8_t _var_1326;
  generic8_t _var_1327;
  generic8_t _var_1328;
  generic8_t _var_1329;
  generic64_t _var_1330;
  generic16_t _var_1331;
  generic64_t _var_1332;
  generic16_t _var_1333;
  generic64_t _var_1334;
  generic16_t _var_1335;
  generic64_t _var_1336;
  generic16_t _var_1337;
  generic64_t _var_1338;
  generic16_t _var_1339;
  generic64_t _var_1340;
  generic16_t _var_1341;
  generic64_t _var_1342;
  generic16_t _var_1343;
  generic64_t _var_1344;
  generic16_t _var_1345;
  generic8_t _var_1346[7528];
  generic32_t _var_1347;
  generic32_t _var_1348;
  generic64_t _var_1349;
  generic64_t _var_1350;
  generic64_t _var_1351;
  generic64_t _var_1352;
  generic64_t _var_1353;
  generic64_t _var_1354;
  generic8_t _var_1355;
  generic64_t _var_1356;
  generic32_t _var_1357;
  generic64_t _var_1358;
  generic64_t _var_1359;
  generic32_t _var_1360;
  generic64_t _var_1361;
  generic64_t _var_1362;
  generic64_t _var_1363;
  generic32_t _var_1364;
  generic16_t _var_1365;
  generic64_t _var_1366;
  generic16_t _var_1367;
  generic64_t _var_1368;
  generic16_t _var_1369;
  generic64_t _var_1370;
  generic16_t _var_1371;
  generic64_t _var_1372;
  generic16_t _var_1373;
  generic64_t _var_1374;
  generic16_t _var_1375;
  generic64_t _var_1376;
  generic16_t _var_1377;
  generic64_t _var_1378;
  generic16_t _var_1379;
  generic64_t _var_1380;
  generic16_t _var_1381;
  generic8_t _var_1382;
  generic64_t _var_1383;
  generic32_t _var_1384;
  generic64_t _var_1385;
  generic16_t _var_1386;
  generic64_t _var_1387;
  generic16_t _var_1388;
  generic64_t _var_1389;
  generic16_t _var_1390;
  generic64_t _var_1391;
  generic16_t _var_1392;
  generic64_t _var_1393;
  generic16_t _var_1394;
  generic64_t _var_1395;
  generic16_t _var_1396;
  generic64_t _var_1397;
  generic16_t _var_1398;
  generic64_t _var_1399;
  generic16_t _var_1400;
  generic8_t _var_1401;
  generic64_t _var_1402;
  void *_var_1403;
  void *_var_1404;
  void *_var_1405;
  void *_var_1406;
  void *_var_1407;
  void *_var_1408;
  void *_var_1409;
  void *_var_1410;
  void *_var_1411;
  void *_var_1412;
  void *_var_1413;
  void *_var_1414;
  void *_var_1415;
  void *_var_1416;
  void *_var_1417;
  void *_var_1418;
  void *_var_1419;
  generic64_t _var_1420;
  generic64_t _var_1421;
  generic64_t _var_1422;
  generic64_t _var_1423;
  generic64_t _var_1424;
  generic64_t _var_1425;
  generic64_t _var_1426;
  generic64_t _var_1427;
  generic64_t _var_1428;
  generic64_t _var_1429;
  generic64_t _var_1430;
  generic64_t _var_1431;
  generic64_t _var_1432;
  generic64_t _var_1433;
  generic32_t _var_1434;
  generic64_t _var_1435;
  generic64_t _var_1436;
  generic64_t _var_1437;
  generic64_t _var_1438;
  generic64_t _var_1439;
  generic64_t _var_1440;
  generic64_t _var_1441;
  generic64_t _var_1442;
  generic64_t _var_1443;
  generic64_t _var_1444;
  generic32_t _var_1445;
  generic64_t _var_1446;
  generic64_t _var_1447;
  generic64_t _var_1448;
  generic64_t _var_1449;
  generic64_t _var_1450;
  generic64_t _var_1451;
  generic64_t _var_1452;
  generic64_t _var_1453;
  generic64_t _var_1454;
  generic64_t _var_1455;
  generic32_t _var_1456;
  generic64_t _var_1457;
  generic64_t _var_1458;
  generic64_t _var_1459;
  generic64_t _var_1460;
  generic64_t _var_1461;
  generic64_t _var_1462;
  generic64_t _var_1463;
  generic16_t _var_1464;
  generic64_t _var_1465;
  generic16_t _var_1466;
  generic64_t _var_1467;
  generic16_t _var_1468;
  generic64_t _var_1469;
  generic16_t _var_1470;
  generic64_t _var_1471;
  generic16_t _var_1472;
  generic64_t _var_1473;
  generic16_t _var_1474;
  generic64_t _var_1475;
  generic16_t _var_1476;
  generic64_t _var_1477;
  generic16_t _var_1478;
  generic32_t _var_1479;
  generic64_t _var_1480;
  generic16_t _var_1481;
  generic64_t _var_1482;
  generic16_t _var_1483;
  generic64_t _var_1484;
  generic16_t _var_1485;
  generic64_t _var_1486;
  generic16_t _var_1487;
  generic64_t _var_1488;
  generic16_t _var_1489;
  generic64_t _var_1490;
  generic16_t _var_1491;
  generic64_t _var_1492;
  generic16_t _var_1493;
  generic64_t _var_1494;
  generic16_t _var_1495;
  generic8_t _var_1496;
  generic32_t _var_1497;
  void *_var_1498;
  generic8_t _var_1499;
  generic64_t _var_1500;
  generic64_t _var_1501;
  generic64_t _var_1502;
  generic64_t _var_1503;
  generic64_t _var_1504;
  generic32_t _var_1505;
  generic64_t _var_1506;
  generic64_t _var_1507;
  generic32_t _var_1508;
  generic64_t _var_1509;
  generic64_t _var_1510;
  generic64_t _var_1511;
  generic64_t _var_1512;
  generic64_t _var_1513;
  generic32_t _var_1514;
  generic32_t _var_1515;
  generic32_t _var_1516;
  generic64_t _var_1517;
  generic32_t _var_1518;
  generic64_t _var_1519;
  generic64_t _var_1520;
  generic64_t _var_1521;
  generic64_t _var_1522;
  generic32_t _var_1523;
  generic64_t _var_1524;
  generic64_t _var_1525;
  generic64_t _var_1526;
  generic64_t _var_1527;
  generic64_t _var_1528;
  generic64_t _var_1529;
  generic64_t _var_1530;
  generic64_t _var_1531;
  generic32_t _var_1532;
  generic64_t _var_1533;
  generic64_t _var_1534;
  generic64_t _var_1535;
  generic64_t _var_1536;
  generic32_t _var_1537;
  generic16_t _var_1538;
  generic64_t _var_1539;
  generic16_t _var_1540;
  generic64_t _var_1541;
  generic16_t _var_1542;
  generic64_t _var_1543;
  generic16_t _var_1544;
  generic64_t _var_1545;
  generic16_t _var_1546;
  generic64_t _var_1547;
  generic16_t _var_1548;
  generic64_t _var_1549;
  generic16_t _var_1550;
  generic64_t _var_1551;
  generic16_t _var_1552;
  generic64_t _var_1553;
  generic16_t _var_1554;
  generic8_t _var_1555;
  generic64_t _var_1556;
  generic16_t _var_1557;
  generic64_t _var_1558;
  generic16_t _var_1559;
  generic64_t _var_1560;
  generic16_t _var_1561;
  generic64_t _var_1562;
  generic16_t _var_1563;
  generic64_t _var_1564;
  generic16_t _var_1565;
  generic64_t _var_1566;
  generic16_t _var_1567;
  generic64_t _var_1568;
  generic16_t _var_1569;
  generic64_t _var_1570;
  generic16_t _var_1571;
  generic8_t _var_1572;
  generic64_t _var_1573;
  generic32_t _var_1574;
  generic64_t _var_1575;
  generic64_t _var_1576;
  generic16_t _var_1577;
  generic64_t _var_1578;
  generic16_t _var_1579;
  generic64_t _var_1580;
  generic16_t _var_1581;
  generic64_t _var_1582;
  generic16_t _var_1583;
  generic64_t _var_1584;
  generic16_t _var_1585;
  generic64_t _var_1586;
  generic16_t _var_1587;
  generic64_t _var_1588;
  generic16_t _var_1589;
  generic64_t _var_1590;
  generic16_t _var_1591;
  generic32_t _var_1592;
  generic32_t _var_1593;
  generic32_t _var_1594;
  generic8_t _var_1595[4];
  generic8_t _var_1596[8];
  generic8_t _var_1597[4];
  generic8_t _var_1598[8];
  generic8_t _var_1599[16];
  generic8_t _var_1600[4];
  generic8_t _var_1601[8];
  generic8_t _var_1602[8];
  generic8_t _var_1603[8];
  generic8_t _var_1604[8];
  generic8_t _var_1605[8];
  generic8_t _var_1606[8];
  generic8_t _var_1607[8];
  generic8_t _var_1608[8];
  generic8_t _var_1609[8];
  generic8_t _var_1610[8];
  generic8_t _var_1611[8];
  generic8_t _var_1612[8];
  generic8_t _var_1613[8];
  generic8_t _var_1614[8];
  generic8_t _var_1615[4];
  generic8_t _var_1616[4];
  generic8_t _var_1617[8];
  generic8_t _var_1618[4];
  generic8_t _var_1619[8];
  generic8_t _var_1620[2];
  generic8_t _var_1621[8];
  generic8_t _var_1622[2];
  generic8_t _var_1623[8];
  generic8_t _var_1624[2];
  generic8_t _var_1625[8];
  generic8_t _var_1626[2];
  generic8_t _var_1627[8];
  generic8_t _var_1628[2];
  generic8_t _var_1629[8];
  generic8_t _var_1630[2];
  generic8_t _var_1631[8];
  generic8_t _var_1632[2];
  generic8_t _var_1633[8];
  generic8_t _var_1634[2];
  generic8_t _var_1635[8];
  generic8_t _var_1636[2];
  generic8_t _var_1637[8];
  generic8_t _var_1638[2];
  generic8_t _var_1639[8];
  generic8_t _var_1640[2];
  generic8_t _var_1641[8];
  generic8_t _var_1642[2];
  generic8_t _var_1643[8];
  generic8_t _var_1644[2];
  generic8_t _var_1645[8];
  generic8_t _var_1646[2];
  generic8_t _var_1647[8];
  generic8_t _var_1648[2];
  generic8_t _var_1649[8];
  generic8_t _var_1650[2];
  generic8_t _var_1651[4];
  generic8_t _var_1652[4];
  generic8_t _var_1653[8];
  generic8_t _var_1654[2];
  generic8_t _var_1655[8];
  generic8_t _var_1656[2];
  generic8_t _var_1657[8];
  generic8_t _var_1658[2];
  generic8_t _var_1659[8];
  generic8_t _var_1660[2];
  generic8_t _var_1661[8];
  generic8_t _var_1662[2];
  generic8_t _var_1663[8];
  generic8_t _var_1664[2];
  generic8_t _var_1665[8];
  generic8_t _var_1666[2];
  generic8_t _var_1667[8];
  generic8_t _var_1668[2];
  generic8_t _var_1669[4];
  generic8_t _var_1670[8];
  generic8_t _var_1671[8];
  generic8_t _var_1672[2];
  generic8_t _var_1673[8];
  generic8_t _var_1674[2];
  generic8_t _var_1675[8];
  generic8_t _var_1676[2];
  generic8_t _var_1677[8];
  generic8_t _var_1678[2];
  generic8_t _var_1679[8];
  generic8_t _var_1680[2];
  generic8_t _var_1681[8];
  generic8_t _var_1682[2];
  generic8_t _var_1683[8];
  generic8_t _var_1684[2];
  generic8_t _var_1685[8];
  generic8_t _var_1686[2];
  generic8_t _var_1687[4];
  generic8_t _var_1688[4];
  generic8_t _var_1689[4];
  generic8_t _var_1690[4];
  generic8_t _var_1691[8];
  generic8_t _var_1692[2];
  generic8_t _var_1693[8];
  generic8_t _var_1694[2];
  generic8_t _var_1695[8];
  generic8_t _var_1696[2];
  generic8_t _var_1697[8];
  generic8_t _var_1698[2];
  generic8_t _var_1699[8];
  generic8_t _var_1700[2];
  generic8_t _var_1701[8];
  generic8_t _var_1702[2];
  generic8_t _var_1703[8];
  generic8_t _var_1704[2];
  generic8_t _var_1705[8];
  generic8_t _var_1706[2];
  generic8_t _var_1707[8];
  generic8_t _var_1708;
  generic8_t _var_1709[8];
  generic8_t _var_1710[4];
  generic8_t _var_1711[8];
  generic8_t _var_1712[2];
  generic8_t _var_1713[8];
  generic8_t _var_1714[2];
  generic8_t _var_1715[8];
  generic8_t _var_1716[2];
  generic8_t _var_1717[8];
  generic8_t _var_1718[2];
  generic8_t _var_1719[8];
  generic8_t _var_1720[2];
  generic8_t _var_1721[8];
  generic8_t _var_1722[2];
  generic8_t _var_1723[8];
  generic8_t _var_1724[2];
  generic8_t _var_1725[8];
  generic8_t _var_1726[2];
  generic8_t _var_1727;
  generic8_t _var_1728[4];
  generic8_t _var_1729[4];
  generic8_t _var_1730[8];
  generic8_t _var_1731[8];
  generic8_t _var_1732[4];
  generic8_t _var_1733[2];
  generic8_t _var_1734;
  generic8_t _var_1735[4];
  generic8_t _var_1736;
  generic8_t _var_1737;
  generic8_t _var_1738[8];
  generic8_t _var_1739;
  generic8_t _var_1740[4];
  generic8_t _var_1741[8];
  generic8_t _var_1742[4];
  generic8_t _var_1743[8];
  generic8_t _var_1744[8];
  generic8_t _var_1745[8];
  generic8_t _var_1746[8];
  generic8_t _var_1747[8];
  generic8_t _var_1748;
  generic8_t _var_1749[8];
  generic8_t _var_1750[8];
  generic8_t _var_1751[8];
  generic8_t _var_1752[4];
  generic8_t _var_1753[4];
  generic8_t _var_1754[4];
  generic8_t _var_1755[8];
  generic8_t _var_1756[8];
  generic8_t _var_1757[4];
  generic8_t _var_1758[8];
  generic8_t _var_1759[8];
  generic8_t _var_1760[4];
  generic8_t _var_1761[8];
  generic8_t _var_1762[8];
  generic8_t _var_1763[8];
  generic8_t _var_1764[4];
  generic8_t _var_1765[8];
  generic8_t _var_1766[8];
  generic8_t _var_1767[8];
  generic8_t _var_1768;
  generic8_t _var_1769[8];
  generic8_t _var_1770[8];
  generic8_t _var_1771[8];
  generic8_t _var_1772[8];
  generic8_t _var_1773[8];
  generic8_t _var_1774[4];
  generic8_t _var_1775[8];
  generic8_t _var_1776[8];
  generic8_t _var_1777[8];
  generic8_t _var_1778;
  generic8_t _var_1779[8];
  generic8_t _var_1780;
  generic8_t _var_1781[4];
  generic8_t _var_1782[8];
  generic8_t _var_1783[8];
  generic8_t _var_1784[8];
  generic8_t _var_1785[8];
  generic8_t _var_1786[8];
  generic8_t _var_1787;
  generic8_t _var_1788[4];
  generic8_t _var_1789[4];
  generic8_t _var_1790[8];
  generic8_t _var_1791[4];
  generic8_t _var_1792[8];
  generic8_t _var_1793[8];
  generic8_t _var_1794[8];
  generic8_t _var_1795[8];
  generic8_t _var_1796[8];
  generic8_t _var_1797[8];
  generic8_t _var_1798[8];
  generic8_t _var_1799[8];
  generic8_t _var_1800[8];
  generic8_t _var_1801[8];
  generic8_t _var_1802;
  generic8_t _var_1803[8];
  generic8_t _var_1804[8];
  generic8_t _var_1805[4];
  generic8_t _var_1806[8];
  generic8_t _var_1807[8];
  generic8_t _var_1808[8];
  generic8_t _var_1809[4];
  generic8_t _var_1810[8];
  generic8_t _var_1811[2];
  generic8_t _var_1812[8];
  generic8_t _var_1813[2];
  generic8_t _var_1814[8];
  generic8_t _var_1815[2];
  generic8_t _var_1816[8];
  generic8_t _var_1817[2];
  generic8_t _var_1818[8];
  generic8_t _var_1819[2];
  generic8_t _var_1820[8];
  generic8_t _var_1821[2];
  generic8_t _var_1822[8];
  generic8_t _var_1823[2];
  generic8_t _var_1824[8];
  generic8_t _var_1825[2];
  generic8_t _var_1826;
  generic8_t _var_1827;
  generic8_t _var_1828[2];
  generic8_t _var_1829[8];
  generic8_t _var_1830[2];
  generic8_t _var_1831[8];
  generic8_t _var_1832[2];
  generic8_t _var_1833[8];
  generic8_t _var_1834[2];
  generic8_t _var_1835[8];
  generic8_t _var_1836[2];
  generic8_t _var_1837[8];
  generic8_t _var_1838[2];
  generic8_t _var_1839[8];
  generic8_t _var_1840[2];
  generic8_t _var_1841[8];
  generic8_t _var_1842[2];
  generic8_t _var_1843[8];
  generic8_t _var_1844;
  generic8_t _var_1845[4];
  generic8_t _var_1846[8];
  generic8_t _var_1847[2];
  generic8_t _var_1848[8];
  generic8_t _var_1849[2];
  generic8_t _var_1850[8];
  generic8_t _var_1851[2];
  generic8_t _var_1852[8];
  generic8_t _var_1853[2];
  generic8_t _var_1854[8];
  generic8_t _var_1855[2];
  generic8_t _var_1856[8];
  generic8_t _var_1857[2];
  generic8_t _var_1858[8];
  generic8_t _var_1859[2];
  generic8_t _var_1860[8];
  generic8_t _var_1861[2];
  generic8_t _var_1862[8];
  generic8_t _var_1863[4];
  generic8_t _var_1864;
  generic8_t _var_1865[4];
  generic8_t _var_1866[8];
  generic8_t _var_1867[8];
  generic8_t _var_1868[4];
  generic8_t _var_1869[8];
  generic8_t _var_1870[2];
  generic8_t _var_1871[8];
  generic8_t _var_1872[2];
  generic8_t _var_1873[8];
  generic8_t _var_1874[2];
  generic8_t _var_1875[8];
  generic8_t _var_1876[2];
  generic8_t _var_1877[8];
  generic8_t _var_1878[2];
  generic8_t _var_1879[8];
  generic8_t _var_1880[2];
  generic8_t _var_1881[8];
  generic8_t _var_1882[2];
  generic8_t _var_1883[8];
  generic8_t _var_1884[2];
  generic8_t _var_1885[2];
  generic8_t _var_1886;
  generic8_t _var_1887[4];
  generic8_t _var_1888[2];
  generic8_t _var_1889;
  generic8_t _var_1890[8];
  generic8_t _var_1891[2];
  generic8_t _var_1892[8];
  generic8_t _var_1893[2];
  generic8_t _var_1894[8];
  generic8_t _var_1895[2];
  generic8_t _var_1896[8];
  generic8_t _var_1897[2];
  generic8_t _var_1898[8];
  generic8_t _var_1899[2];
  generic8_t _var_1900[8];
  generic8_t _var_1901[2];
  generic8_t _var_1902[8];
  generic8_t _var_1903[2];
  generic8_t _var_1904[8];
  generic8_t _var_1905[2];
  generic8_t _var_1906;
  generic8_t _var_1907;
  generic8_t _var_1908[8];
  generic8_t _var_1909[8];
  generic8_t _var_1910[4];
  generic8_t _var_1911[4];
  generic8_t _var_1912[4];
  helper_fldt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346 + 7536ul, 0u, (void *)&_var_1321, (void *)&_var_1322, (void *)&_var_1323, (void *)&_var_1324, (void *)&_var_1325, (void *)&_var_1326, (void *)&_var_1327, (void *)&_var_1328, (void *)&_var_1329, (void *)&_var_1330, (void *)&_var_1331, (void *)&_var_1332, (void *)&_var_1333, (void *)&_var_1334, (void *)&_var_1335, (void *)&_var_1336, (void *)&_var_1337, (void *)&_var_1338, (void *)&_var_1339, (void *)&_var_1340, (void *)&_var_1341, (void *)&_var_1342, (void *)&_var_1343, (void *)&_var_1344, (void *)&_var_1345);
  *(generic32_t *)&_var_1618 = _var_1321;
  *(generic64_t *)&_var_1619 = _var_1330;
  *(generic16_t *)&_var_1620 = _var_1331;
  *(generic64_t *)&_var_1621 = _var_1332;
  *(generic16_t *)&_var_1622 = _var_1333;
  *(generic64_t *)&_var_1623 = _var_1334;
  *(generic16_t *)&_var_1624 = _var_1335;
  *(generic64_t *)&_var_1625 = _var_1336;
  *(generic16_t *)&_var_1626 = _var_1337;
  *(generic64_t *)&_var_1627 = _var_1338;
  *(generic16_t *)&_var_1628 = _var_1339;
  *(generic64_t *)&_var_1629 = _var_1340;
  *(generic16_t *)&_var_1630 = _var_1341;
  *(generic64_t *)&_var_1631 = _var_1342;
  *(generic16_t *)&_var_1632 = _var_1343;
  *(generic64_t *)&_var_1633 = _var_1344;
  *(generic16_t *)&_var_1634 = _var_1345;
  ((generic32_t *)&_var_1346)[4ul] = (generic32_t)w;
  ((generic32_t *)&_var_1346)[5ul] = (generic32_t)fl;
  ((generic32_t *)&_var_1346)[6ul] = (generic32_t)t;
  helper_fpush_wrapper((void *)0ul, *(generic32_t *)&_var_1618, (void *)&_var_1312, (void *)&_var_1313, (void *)&_var_1314, (void *)&_var_1315, (void *)&_var_1316, (void *)&_var_1317, (void *)&_var_1318, (void *)&_var_1319, (void *)&_var_1320);
  helper_fmov_ST0_STN_wrapper((void *)0ul, 1u, _var_1312, *(generic64_t *)&_var_1619, *(generic16_t *)&_var_1620, *(generic64_t *)&_var_1621, *(generic16_t *)&_var_1622, *(generic64_t *)&_var_1623, *(generic16_t *)&_var_1624, *(generic64_t *)&_var_1625, *(generic16_t *)&_var_1626, *(generic64_t *)&_var_1627, *(generic16_t *)&_var_1628, *(generic64_t *)&_var_1629, *(generic16_t *)&_var_1630, *(generic64_t *)&_var_1631, *(generic16_t *)&_var_1632, *(generic64_t *)&_var_1633, *(generic16_t *)&_var_1634, (void *)&_var_1296, (void *)&_var_1297, (void *)&_var_1298, (void *)&_var_1299, (void *)&_var_1300, (void *)&_var_1301, (void *)&_var_1302, (void *)&_var_1303, (void *)&_var_1304, (void *)&_var_1305, (void *)&_var_1306, (void *)&_var_1307, (void *)&_var_1308, (void *)&_var_1309, (void *)&_var_1310, (void *)&_var_1311);
  *(generic64_t *)&_var_1635 = _var_1296;
  *(generic16_t *)&_var_1636 = _var_1297;
  *(generic64_t *)&_var_1637 = _var_1298;
  *(generic16_t *)&_var_1638 = _var_1299;
  *(generic64_t *)&_var_1639 = _var_1300;
  *(generic16_t *)&_var_1640 = _var_1301;
  *(generic64_t *)&_var_1641 = _var_1302;
  *(generic16_t *)&_var_1642 = _var_1303;
  *(generic64_t *)&_var_1643 = _var_1304;
  *(generic16_t *)&_var_1644 = _var_1305;
  *(generic64_t *)&_var_1645 = _var_1306;
  *(generic16_t *)&_var_1646 = _var_1307;
  *(generic64_t *)&_var_1647 = _var_1308;
  *(generic16_t *)&_var_1648 = _var_1309;
  *(generic64_t *)&_var_1649 = _var_1310;
  *(generic16_t *)&_var_1650 = _var_1311;
  helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346, _var_1312, *(generic64_t *)&_var_1635, *(generic16_t *)&_var_1636, *(generic64_t *)&_var_1637, *(generic16_t *)&_var_1638, *(generic64_t *)&_var_1639, *(generic16_t *)&_var_1640, *(generic64_t *)&_var_1641, *(generic16_t *)&_var_1642, *(generic64_t *)&_var_1643, *(generic16_t *)&_var_1644, *(generic64_t *)&_var_1645, *(generic16_t *)&_var_1646, *(generic64_t *)&_var_1647, *(generic16_t *)&_var_1648, *(generic64_t *)&_var_1649, *(generic16_t *)&_var_1650);
  helper_fpop_wrapper((void *)0ul, _var_1312, (void *)&_var_1287, (void *)&_var_1288, (void *)&_var_1289, (void *)&_var_1290, (void *)&_var_1291, (void *)&_var_1292, (void *)&_var_1293, (void *)&_var_1294, (void *)&_var_1295);
  *(generic32_t *)&_var_1651 = _var_1287;
  ((generic32_t *)&_var_1346)[22ul] = 0u;
  helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346 + 7536ul, *(generic32_t *)&_var_1651, *(generic64_t *)&_var_1635, *(generic16_t *)&_var_1636, *(generic64_t *)&_var_1637, *(generic16_t *)&_var_1638, *(generic64_t *)&_var_1639, *(generic16_t *)&_var_1640, *(generic64_t *)&_var_1641, *(generic16_t *)&_var_1642, *(generic64_t *)&_var_1643, *(generic16_t *)&_var_1644, *(generic64_t *)&_var_1645, *(generic16_t *)&_var_1646, *(generic64_t *)&_var_1647, *(generic16_t *)&_var_1648, *(generic64_t *)&_var_1649, *(generic16_t *)&_var_1650);
  helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1651, (void *)&_var_1278, (void *)&_var_1279, (void *)&_var_1280, (void *)&_var_1281, (void *)&_var_1282, (void *)&_var_1283, (void *)&_var_1284, (void *)&_var_1285, (void *)&_var_1286);
  *(int32_t *)&_var_1595 = unreserved___signbitl((float128_t)((generic128_t)y & (generic128_t)18446744073709551615u));
  helper_fldt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346 + 7536ul, _var_1278, (void *)&_var_1253, (void *)&_var_1254, (void *)&_var_1255, (void *)&_var_1256, (void *)&_var_1257, (void *)&_var_1258, (void *)&_var_1259, (void *)&_var_1260, (void *)&_var_1261, (void *)&_var_1262, (void *)&_var_1263, (void *)&_var_1264, (void *)&_var_1265, (void *)&_var_1266, (void *)&_var_1267, (void *)&_var_1268, (void *)&_var_1269, (void *)&_var_1270, (void *)&_var_1271, (void *)&_var_1272, (void *)&_var_1273, (void *)&_var_1274, (void *)&_var_1275, (void *)&_var_1276, (void *)&_var_1277);
  *(generic32_t *)&_var_1652 = _var_1253;
  *(generic64_t *)&_var_1653 = _var_1262;
  *(generic16_t *)&_var_1654 = _var_1263;
  *(generic64_t *)&_var_1655 = _var_1264;
  *(generic16_t *)&_var_1656 = _var_1265;
  *(generic64_t *)&_var_1657 = _var_1266;
  *(generic16_t *)&_var_1658 = _var_1267;
  *(generic64_t *)&_var_1659 = _var_1268;
  *(generic16_t *)&_var_1660 = _var_1269;
  *(generic64_t *)&_var_1661 = _var_1270;
  *(generic16_t *)&_var_1662 = _var_1271;
  *(generic64_t *)&_var_1663 = _var_1272;
  *(generic16_t *)&_var_1664 = _var_1273;
  *(generic64_t *)&_var_1665 = _var_1274;
  *(generic16_t *)&_var_1666 = _var_1275;
  *(generic64_t *)&_var_1667 = _var_1276;
  *(generic16_t *)&_var_1668 = _var_1277;
  if (*(generic32_t *)&_var_1595 == 0u) {
    *(generic64_t *)&_var_1596 = lshift(0ul, 4294967272u);
    *(generic32_t *)&_var_1669 = (generic32_t)*(generic64_t *)&_var_1596 & 128u | (generic32_t)((uint32_t)((generic32_t *)&_var_1346)[5ul] >> 11u) & 1u | 68u;
    if ((((generic32_t *)&_var_1346)[5ul] & 2048u) == 0u) {
      _var_1594 = ((generic32_t *)&_var_1346)[5ul] & 1u;
      ((generic32_t *)&_var_1346)[12ul] = _var_1594;
      _var_1575 = _var_1594 == 0u ? 4215921ul : 4215926ul;
      _var_1576 = *(generic64_t *)&_var_1653;
      _var_1577 = *(generic16_t *)&_var_1654;
      _var_1578 = *(generic64_t *)&_var_1655;
      _var_1579 = *(generic16_t *)&_var_1656;
      _var_1580 = *(generic64_t *)&_var_1657;
      _var_1581 = *(generic16_t *)&_var_1658;
      _var_1582 = *(generic64_t *)&_var_1659;
      _var_1583 = *(generic16_t *)&_var_1660;
      _var_1584 = *(generic64_t *)&_var_1661;
      _var_1585 = *(generic16_t *)&_var_1662;
      _var_1586 = *(generic64_t *)&_var_1663;
      _var_1587 = *(generic16_t *)&_var_1664;
      _var_1588 = *(generic64_t *)&_var_1665;
      _var_1589 = *(generic16_t *)&_var_1666;
      _var_1590 = *(generic64_t *)&_var_1667;
      _var_1591 = *(generic16_t *)&_var_1668;
      _var_1592 = *(generic32_t *)&_var_1669;
      _var_1593 = 24u;
    } else {
      ((generic32_t *)&_var_1346)[12ul] = 1u;
      _var_1575 = 4215923ul;
      _var_1576 = *(generic64_t *)&_var_1653;
      _var_1577 = *(generic16_t *)&_var_1654;
      _var_1578 = *(generic64_t *)&_var_1655;
      _var_1579 = *(generic16_t *)&_var_1656;
      _var_1580 = *(generic64_t *)&_var_1657;
      _var_1581 = *(generic16_t *)&_var_1658;
      _var_1582 = *(generic64_t *)&_var_1659;
      _var_1583 = *(generic16_t *)&_var_1660;
      _var_1584 = *(generic64_t *)&_var_1661;
      _var_1585 = *(generic16_t *)&_var_1662;
      _var_1586 = *(generic64_t *)&_var_1663;
      _var_1587 = *(generic16_t *)&_var_1664;
      _var_1588 = *(generic64_t *)&_var_1665;
      _var_1589 = *(generic16_t *)&_var_1666;
      _var_1590 = *(generic64_t *)&_var_1667;
      _var_1591 = *(generic16_t *)&_var_1668;
      _var_1592 = *(generic32_t *)&_var_1669;
      _var_1593 = 1u;
      _var_1594 = *(generic32_t *)&_var_1595;
    }
  } else {
    ((generic32_t *)&_var_1346)[12ul] = 1u;
    helper_fchs_ST0_wrapper((void *)0ul, *(generic32_t *)&_var_1652, *(generic64_t *)&_var_1653, *(generic16_t *)&_var_1654, *(generic64_t *)&_var_1655, *(generic16_t *)&_var_1656, *(generic64_t *)&_var_1657, *(generic16_t *)&_var_1658, *(generic64_t *)&_var_1659, *(generic16_t *)&_var_1660, *(generic64_t *)&_var_1661, *(generic16_t *)&_var_1662, *(generic64_t *)&_var_1663, *(generic16_t *)&_var_1664, *(generic64_t *)&_var_1665, *(generic16_t *)&_var_1666, *(generic64_t *)&_var_1667, *(generic16_t *)&_var_1668, (void *)&_var_1237, (void *)&_var_1238, (void *)&_var_1239, (void *)&_var_1240, (void *)&_var_1241, (void *)&_var_1242, (void *)&_var_1243, (void *)&_var_1244, (void *)&_var_1245, (void *)&_var_1246, (void *)&_var_1247, (void *)&_var_1248, (void *)&_var_1249, (void *)&_var_1250, (void *)&_var_1251, (void *)&_var_1252);
    _var_1576 = _var_1237;
    _var_1577 = _var_1238;
    _var_1578 = _var_1239;
    _var_1579 = _var_1240;
    _var_1580 = _var_1241;
    _var_1581 = _var_1242;
    _var_1582 = _var_1243;
    _var_1583 = _var_1244;
    _var_1584 = _var_1245;
    _var_1585 = _var_1246;
    _var_1586 = _var_1247;
    _var_1587 = _var_1248;
    _var_1588 = _var_1249;
    _var_1589 = _var_1250;
    _var_1590 = _var_1251;
    _var_1591 = _var_1252;
    _var_1575 = 4215920ul;
    _var_1592 = 7480u;
    _var_1593 = 24u;
    _var_1594 = *(generic32_t *)&_var_1595;
  }
  *(generic64_t *)&_var_1670 = _var_1575;
  *(generic64_t *)&_var_1671 = _var_1576;
  *(generic16_t *)&_var_1672 = _var_1577;
  *(generic64_t *)&_var_1673 = _var_1578;
  *(generic16_t *)&_var_1674 = _var_1579;
  *(generic64_t *)&_var_1675 = _var_1580;
  *(generic16_t *)&_var_1676 = _var_1581;
  *(generic64_t *)&_var_1677 = _var_1582;
  *(generic16_t *)&_var_1678 = _var_1583;
  *(generic64_t *)&_var_1679 = _var_1584;
  *(generic16_t *)&_var_1680 = _var_1585;
  *(generic64_t *)&_var_1681 = _var_1586;
  *(generic16_t *)&_var_1682 = _var_1587;
  *(generic64_t *)&_var_1683 = _var_1588;
  *(generic16_t *)&_var_1684 = _var_1589;
  *(generic64_t *)&_var_1685 = _var_1590;
  *(generic16_t *)&_var_1686 = _var_1591;
  *(generic32_t *)&_var_1687 = _var_1592;
  *(generic32_t *)&_var_1688 = _var_1593;
  *(generic32_t *)&_var_1689 = _var_1594;
  ((generic64_t *)&_var_1346)[1ul] = (generic64_t)(uint64_t)(uint32_t)w;
  *(generic64_t *)&_var_1346 = (generic64_t)(uint64_t)(uint32_t)w;
  helper_fpush_wrapper((void *)0ul, *(generic32_t *)&_var_1652, (void *)&_var_1228, (void *)&_var_1229, (void *)&_var_1230, (void *)&_var_1231, (void *)&_var_1232, (void *)&_var_1233, (void *)&_var_1234, (void *)&_var_1235, (void *)&_var_1236);
  helper_fmov_ST0_STN_wrapper((void *)0ul, 1u, _var_1228, *(generic64_t *)&_var_1671, *(generic16_t *)&_var_1672, *(generic64_t *)&_var_1673, *(generic16_t *)&_var_1674, *(generic64_t *)&_var_1675, *(generic16_t *)&_var_1676, *(generic64_t *)&_var_1677, *(generic16_t *)&_var_1678, *(generic64_t *)&_var_1679, *(generic16_t *)&_var_1680, *(generic64_t *)&_var_1681, *(generic16_t *)&_var_1682, *(generic64_t *)&_var_1683, *(generic16_t *)&_var_1684, *(generic64_t *)&_var_1685, *(generic16_t *)&_var_1686, (void *)&_var_1212, (void *)&_var_1213, (void *)&_var_1214, (void *)&_var_1215, (void *)&_var_1216, (void *)&_var_1217, (void *)&_var_1218, (void *)&_var_1219, (void *)&_var_1220, (void *)&_var_1221, (void *)&_var_1222, (void *)&_var_1223, (void *)&_var_1224, (void *)&_var_1225, (void *)&_var_1226, (void *)&_var_1227);
  helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346, _var_1228, _var_1212, _var_1213, _var_1214, _var_1215, _var_1216, _var_1217, _var_1218, _var_1219, _var_1220, _var_1221, _var_1222, _var_1223, _var_1224, _var_1225, _var_1226, _var_1227);
  helper_fpop_wrapper((void *)0ul, _var_1228, (void *)&_var_1203, (void *)&_var_1204, (void *)&_var_1205, (void *)&_var_1206, (void *)&_var_1207, (void *)&_var_1208, (void *)&_var_1209, (void *)&_var_1210, (void *)&_var_1211);
  helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346 + 7536ul, _var_1203, _var_1212, _var_1213, _var_1214, _var_1215, _var_1216, _var_1217, _var_1218, _var_1219, _var_1220, _var_1221, _var_1222, _var_1223, _var_1224, _var_1225, _var_1226, _var_1227);
  helper_fpop_wrapper((void *)0ul, _var_1203, (void *)&_var_1194, (void *)&_var_1195, (void *)&_var_1196, (void *)&_var_1197, (void *)&_var_1198, (void *)&_var_1199, (void *)&_var_1200, (void *)&_var_1201, (void *)&_var_1202);
  *(int32_t *)&_var_1597 = unreserved___fpclassifyl((float128_t)((generic128_t)y & (generic128_t)18446744073709551615u));
  helper_fldt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346 + 7536ul, _var_1194, (void *)&_var_1169, (void *)&_var_1170, (void *)&_var_1171, (void *)&_var_1172, (void *)&_var_1173, (void *)&_var_1174, (void *)&_var_1175, (void *)&_var_1176, (void *)&_var_1177, (void *)&_var_1178, (void *)&_var_1179, (void *)&_var_1180, (void *)&_var_1181, (void *)&_var_1182, (void *)&_var_1183, (void *)&_var_1184, (void *)&_var_1185, (void *)&_var_1186, (void *)&_var_1187, (void *)&_var_1188, (void *)&_var_1189, (void *)&_var_1190, (void *)&_var_1191, (void *)&_var_1192, (void *)&_var_1193);
  *(generic32_t *)&_var_1690 = _var_1169;
  *(generic64_t *)&_var_1691 = _var_1178;
  *(generic16_t *)&_var_1692 = _var_1179;
  *(generic64_t *)&_var_1693 = _var_1180;
  *(generic16_t *)&_var_1694 = _var_1181;
  *(generic64_t *)&_var_1695 = _var_1182;
  *(generic16_t *)&_var_1696 = _var_1183;
  *(generic64_t *)&_var_1697 = _var_1184;
  *(generic16_t *)&_var_1698 = _var_1185;
  *(generic64_t *)&_var_1699 = _var_1186;
  *(generic16_t *)&_var_1700 = _var_1187;
  *(generic64_t *)&_var_1701 = _var_1188;
  *(generic16_t *)&_var_1702 = _var_1189;
  *(generic64_t *)&_var_1703 = _var_1190;
  *(generic16_t *)&_var_1704 = _var_1191;
  *(generic64_t *)&_var_1705 = _var_1192;
  *(generic16_t *)&_var_1706 = _var_1193;
  _var_1574 = 0u;
  switch (*(generic32_t *)&_var_1688) {
    case 9u:
      _var_1574 = (generic32_t)(uint32_t)(uint8_t)((uint32_t)*(generic32_t *)&_var_1689 < (uint32_t)*(generic32_t *)&_var_1687);
    case 1u:
      _var_1574 = *(generic32_t *)&_var_1687 & 1u;
    case 16u:
      _var_1574 = (generic32_t)(uint32_t)(uint8_t)((uint32_t)*(generic32_t *)&_var_1689 > (uint32_t)(*(generic32_t *)&_var_1687 ^ 4294967295u));
    case 8u:
      _var_1574 = (generic32_t)(uint32_t)(uint8_t)((uint32_t)*(generic32_t *)&_var_1687 > (uint32_t)*(generic32_t *)&_var_1689);
    default: {
    }
  }
  *(generic64_t *)&_var_1598 = lshift((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul & 4294967295ul, 4294967272u);
  *(generic64_t *)&_var_1707 = (generic64_t)(int64_t)(int32_t)((llvm.ctpop.i32((generic32_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) & 255u) << 2u & 4u | _var_1574 | (generic32_t)(uint32_t)(uint8_t)(((generic8_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) + 1u ^ (generic8_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul)) & 16u) | ((generic32_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) == 0u ? 64u : 0u) | (generic32_t)*(generic64_t *)&_var_1598 & 128u | ((generic32_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) == 2147483647u ? 2048u : 0u)) ^ 4u);
  if ((((generic32_t)((uint32_t)(llvm.ctpop.i32((generic32_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) & 255u) << 2u & 4u | _var_1574 | (generic32_t)(uint32_t)(uint8_t)(((generic8_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) + 1u ^ (generic8_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul)) & 16u) | ((generic32_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) == 0u ? 64u : 0u) | (generic32_t)*(generic64_t *)&_var_1598 & 128u | ((generic32_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) == 2147483647u ? 2048u : 0u)) >> 4u) ^ (llvm.ctpop.i32((generic32_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) & 255u) << 2u & 4u | _var_1574 | (generic32_t)(uint32_t)(uint8_t)(((generic8_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) + 1u ^ (generic8_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul)) & 16u) | ((generic32_t)((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul) == 0u ? 64u : 0u) | (generic32_t)*(generic64_t *)&_var_1598 & 128u)) & 192u) == 0u) {
    ((generic64_t *)&_var_1346)[1ul] = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul & 4294967295ul;
    *(generic64_t *)&_var_1346 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul & 4294967295ul;
    helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346, *(generic32_t *)&_var_1690, *(generic64_t *)&_var_1691, *(generic16_t *)&_var_1692, *(generic64_t *)&_var_1693, *(generic16_t *)&_var_1694, *(generic64_t *)&_var_1695, *(generic16_t *)&_var_1696, *(generic64_t *)&_var_1697, *(generic16_t *)&_var_1698, *(generic64_t *)&_var_1699, *(generic16_t *)&_var_1700, *(generic64_t *)&_var_1701, *(generic16_t *)&_var_1702, *(generic64_t *)&_var_1703, *(generic16_t *)&_var_1704, *(generic64_t *)&_var_1705, *(generic16_t *)&_var_1706);
    helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1690, (void *)&_var_1160, (void *)&_var_1161, (void *)&_var_1162, (void *)&_var_1163, (void *)&_var_1164, (void *)&_var_1165, (void *)&_var_1166, (void *)&_var_1167, (void *)&_var_1168);
    *(float128_t *)&_var_1599 = frexpl((float128_t)((generic128_t)y & (generic128_t)18446744073709551615u), (int32_t *)&_var_1346 + 22ul);
    *(generic64_t *)&_var_1709 = ((generic64_t *)&_var_1346)[1ul];
    helper_fmov_FT0_STN_wrapper((void *)0ul, 0u, _var_1160, *(generic64_t *)&_var_1691, *(generic16_t *)&_var_1692, *(generic64_t *)&_var_1693, *(generic16_t *)&_var_1694, *(generic64_t *)&_var_1695, *(generic16_t *)&_var_1696, *(generic64_t *)&_var_1697, *(generic16_t *)&_var_1698, *(generic64_t *)&_var_1699, *(generic16_t *)&_var_1700, *(generic64_t *)&_var_1701, *(generic16_t *)&_var_1702, *(generic64_t *)&_var_1703, *(generic16_t *)&_var_1704, *(generic64_t *)&_var_1705, *(generic16_t *)&_var_1706, (void *)&_var_1145, (void *)&_var_1146);
    helper_fadd_ST0_FT0_wrapper((void *)0ul, _var_1160, *(generic64_t *)&_var_1691, *(generic16_t *)&_var_1692, *(generic64_t *)&_var_1693, *(generic16_t *)&_var_1694, *(generic64_t *)&_var_1695, *(generic16_t *)&_var_1696, *(generic64_t *)&_var_1697, *(generic16_t *)&_var_1698, *(generic64_t *)&_var_1699, *(generic16_t *)&_var_1700, *(generic64_t *)&_var_1701, *(generic16_t *)&_var_1702, *(generic64_t *)&_var_1703, *(generic16_t *)&_var_1704, *(generic64_t *)&_var_1705, *(generic16_t *)&_var_1706, 0u, 0u, 0u, 80u, 0u, 0u, _var_1145, _var_1146, (void *)&_var_1128, (void *)&_var_1129, (void *)&_var_1130, (void *)&_var_1131, (void *)&_var_1132, (void *)&_var_1133, (void *)&_var_1134, (void *)&_var_1135, (void *)&_var_1136, (void *)&_var_1137, (void *)&_var_1138, (void *)&_var_1139, (void *)&_var_1140, (void *)&_var_1141, (void *)&_var_1142, (void *)&_var_1143, (void *)&_var_1144);
    helper_fpush_wrapper((void *)0ul, _var_1160, (void *)&_var_1119, (void *)&_var_1120, (void *)&_var_1121, (void *)&_var_1122, (void *)&_var_1123, (void *)&_var_1124, (void *)&_var_1125, (void *)&_var_1126, (void *)&_var_1127);
    *(generic32_t *)&_var_1710 = _var_1119;
    helper_fldz_ST0_wrapper((void *)0ul, *(generic32_t *)&_var_1710, (void *)&_var_1103, (void *)&_var_1104, (void *)&_var_1105, (void *)&_var_1106, (void *)&_var_1107, (void *)&_var_1108, (void *)&_var_1109, (void *)&_var_1110, (void *)&_var_1111, (void *)&_var_1112, (void *)&_var_1113, (void *)&_var_1114, (void *)&_var_1115, (void *)&_var_1116, (void *)&_var_1117, (void *)&_var_1118);
    helper_fxchg_ST0_STN_wrapper((void *)0ul, 1u, *(generic32_t *)&_var_1710, _var_1103, _var_1104, _var_1105, _var_1106, _var_1107, _var_1108, _var_1109, _var_1110, _var_1111, _var_1112, _var_1113, _var_1114, _var_1115, _var_1116, _var_1117, _var_1118, (void *)&_var_1087, (void *)&_var_1088, (void *)&_var_1089, (void *)&_var_1090, (void *)&_var_1091, (void *)&_var_1092, (void *)&_var_1093, (void *)&_var_1094, (void *)&_var_1095, (void *)&_var_1096, (void *)&_var_1097, (void *)&_var_1098, (void *)&_var_1099, (void *)&_var_1100, (void *)&_var_1101, (void *)&_var_1102);
    *(generic64_t *)&_var_1711 = _var_1087;
    *(generic16_t *)&_var_1712 = _var_1088;
    *(generic64_t *)&_var_1713 = _var_1089;
    *(generic16_t *)&_var_1714 = _var_1090;
    *(generic64_t *)&_var_1715 = _var_1091;
    *(generic16_t *)&_var_1716 = _var_1092;
    *(generic64_t *)&_var_1717 = _var_1093;
    *(generic16_t *)&_var_1718 = _var_1094;
    *(generic64_t *)&_var_1719 = _var_1095;
    *(generic16_t *)&_var_1720 = _var_1096;
    *(generic64_t *)&_var_1721 = _var_1097;
    *(generic16_t *)&_var_1722 = _var_1098;
    *(generic64_t *)&_var_1723 = _var_1099;
    *(generic16_t *)&_var_1724 = _var_1100;
    *(generic64_t *)&_var_1725 = _var_1101;
    *(generic16_t *)&_var_1726 = _var_1102;
    helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, *(generic32_t *)&_var_1710, *(generic64_t *)&_var_1711, *(generic16_t *)&_var_1712, *(generic64_t *)&_var_1713, *(generic16_t *)&_var_1714, *(generic64_t *)&_var_1715, *(generic16_t *)&_var_1716, *(generic64_t *)&_var_1717, *(generic16_t *)&_var_1718, *(generic64_t *)&_var_1719, *(generic16_t *)&_var_1720, *(generic64_t *)&_var_1721, *(generic16_t *)&_var_1722, *(generic64_t *)&_var_1723, *(generic16_t *)&_var_1724, *(generic64_t *)&_var_1725, *(generic16_t *)&_var_1726, (void *)&_var_1085, (void *)&_var_1086);
    helper_fucomi_ST0_FT0_wrapper((void *)0ul, (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1597 + 18446744073709551615ul, 1u, *(generic64_t *)&_var_1707, 0ul, *(generic32_t *)&_var_1710, *(generic64_t *)&_var_1711, *(generic16_t *)&_var_1712, *(generic64_t *)&_var_1713, *(generic16_t *)&_var_1714, *(generic64_t *)&_var_1715, *(generic16_t *)&_var_1716, *(generic64_t *)&_var_1717, *(generic16_t *)&_var_1718, *(generic64_t *)&_var_1719, *(generic16_t *)&_var_1720, *(generic64_t *)&_var_1721, *(generic16_t *)&_var_1722, *(generic64_t *)&_var_1723, *(generic16_t *)&_var_1724, *(generic64_t *)&_var_1725, *(generic16_t *)&_var_1726, _var_1144, _var_1085, _var_1086, (void *)&_var_1083, (void *)&_var_1084);
    _var_1727 = _var_1084;
    if ((_var_1083 & 68ul) == 64ul) {
    } else {
      ((generic32_t *)&_var_1346)[22ul] = ((generic32_t *)&_var_1346)[22ul] + 4294967295u;
    }
    *(generic32_t *)&_var_1728 = ((generic32_t *)&_var_1346)[6ul] | 32u;
    if (*(generic32_t *)&_var_1728 == 97u) {
      helper_fmov_STN_ST0_wrapper((void *)0ul, 1u, *(generic32_t *)&_var_1710, *(generic64_t *)&_var_1711, *(generic16_t *)&_var_1712, *(generic64_t *)&_var_1713, *(generic16_t *)&_var_1714, *(generic64_t *)&_var_1715, *(generic16_t *)&_var_1716, *(generic64_t *)&_var_1717, *(generic16_t *)&_var_1718, *(generic64_t *)&_var_1719, *(generic16_t *)&_var_1720, *(generic64_t *)&_var_1721, *(generic16_t *)&_var_1722, *(generic64_t *)&_var_1723, *(generic16_t *)&_var_1724, *(generic64_t *)&_var_1725, *(generic16_t *)&_var_1726, (void *)&_var_1067, (void *)&_var_1068, (void *)&_var_1069, (void *)&_var_1070, (void *)&_var_1071, (void *)&_var_1072, (void *)&_var_1073, (void *)&_var_1074, (void *)&_var_1075, (void *)&_var_1076, (void *)&_var_1077, (void *)&_var_1078, (void *)&_var_1079, (void *)&_var_1080, (void *)&_var_1081, (void *)&_var_1082);
      helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1710, (void *)&_var_1058, (void *)&_var_1059, (void *)&_var_1060, (void *)&_var_1061, (void *)&_var_1062, (void *)&_var_1063, (void *)&_var_1064, (void *)&_var_1065, (void *)&_var_1066);
      *(generic64_t *)&_var_1808 = (((generic8_t *)&_var_1346)[24ul] & 32u) == 0u ? *(generic64_t *)&_var_1670 : *(generic64_t *)&_var_1670 + 9ul;
      helper_flds_ST0_wrapper((void *)0ul, *(generic32_t *)4215992ul, _var_1058, _var_1727, 0u, 0u, (void *)&_var_1032, (void *)&_var_1033, (void *)&_var_1034, (void *)&_var_1035, (void *)&_var_1036, (void *)&_var_1037, (void *)&_var_1038, (void *)&_var_1039, (void *)&_var_1040, (void *)&_var_1041, (void *)&_var_1042, (void *)&_var_1043, (void *)&_var_1044, (void *)&_var_1045, (void *)&_var_1046, (void *)&_var_1047, (void *)&_var_1048, (void *)&_var_1049, (void *)&_var_1050, (void *)&_var_1051, (void *)&_var_1052, (void *)&_var_1053, (void *)&_var_1054, (void *)&_var_1055, (void *)&_var_1056, (void *)&_var_1057);
      *(generic32_t *)&_var_1809 = _var_1032;
      *(generic64_t *)&_var_1810 = _var_1041;
      *(generic16_t *)&_var_1811 = _var_1042;
      *(generic64_t *)&_var_1812 = _var_1043;
      *(generic16_t *)&_var_1813 = _var_1044;
      *(generic64_t *)&_var_1814 = _var_1045;
      *(generic16_t *)&_var_1815 = _var_1046;
      *(generic64_t *)&_var_1816 = _var_1047;
      *(generic16_t *)&_var_1817 = _var_1048;
      *(generic64_t *)&_var_1818 = _var_1049;
      *(generic16_t *)&_var_1819 = _var_1050;
      *(generic64_t *)&_var_1820 = _var_1051;
      *(generic16_t *)&_var_1821 = _var_1052;
      *(generic64_t *)&_var_1822 = _var_1053;
      *(generic16_t *)&_var_1823 = _var_1054;
      *(generic64_t *)&_var_1824 = _var_1055;
      *(generic16_t *)&_var_1825 = _var_1056;
      _var_1826 = _var_1057;
      ((generic32_t *)&_var_1346)[12ul] = ((generic32_t *)&_var_1346)[12ul] + 2u;
      if ((generic8_t)((uint32_t)p > 14u)) {
        helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, *(generic32_t *)&_var_1809, *(generic64_t *)&_var_1810, *(generic16_t *)&_var_1811, *(generic64_t *)&_var_1812, *(generic16_t *)&_var_1813, *(generic64_t *)&_var_1814, *(generic16_t *)&_var_1815, *(generic64_t *)&_var_1816, *(generic16_t *)&_var_1817, *(generic64_t *)&_var_1818, *(generic16_t *)&_var_1819, *(generic64_t *)&_var_1820, *(generic16_t *)&_var_1821, *(generic64_t *)&_var_1822, *(generic16_t *)&_var_1823, *(generic64_t *)&_var_1824, *(generic16_t *)&_var_1825, (void *)&_var_1012, (void *)&_var_1013, (void *)&_var_1014, (void *)&_var_1015, (void *)&_var_1016, (void *)&_var_1017, (void *)&_var_1018, (void *)&_var_1019, (void *)&_var_1020, (void *)&_var_1021, (void *)&_var_1022, (void *)&_var_1023, (void *)&_var_1024, (void *)&_var_1025, (void *)&_var_1026, (void *)&_var_1027);
        _var_1385 = _var_1012;
        _var_1386 = _var_1013;
        _var_1387 = _var_1014;
        _var_1388 = _var_1015;
        _var_1389 = _var_1016;
        _var_1390 = _var_1017;
        _var_1391 = _var_1018;
        _var_1392 = _var_1019;
        _var_1393 = _var_1020;
        _var_1394 = _var_1021;
        _var_1395 = _var_1022;
        _var_1396 = _var_1023;
        _var_1397 = _var_1024;
        _var_1398 = _var_1025;
        _var_1399 = _var_1026;
        _var_1400 = _var_1027;
        helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1809, (void *)&_var_1003, (void *)&_var_1004, (void *)&_var_1005, (void *)&_var_1006, (void *)&_var_1007, (void *)&_var_1008, (void *)&_var_1009, (void *)&_var_1010, (void *)&_var_1011);
        _var_1384 = _var_1003;
        _var_1401 = _var_1826;
        _var_1844 = _var_1401;
        helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346 + 32ul, _var_1384, _var_1385, _var_1386, _var_1387, _var_1388, _var_1389, _var_1390, _var_1391, _var_1392, _var_1393, _var_1394, _var_1395, _var_1396, _var_1397, _var_1398, _var_1399, _var_1400);
        helper_fpop_wrapper((void *)0ul, _var_1384, (void *)&_var_948, (void *)&_var_949, (void *)&_var_950, (void *)&_var_951, (void *)&_var_952, (void *)&_var_953, (void *)&_var_954, (void *)&_var_955, (void *)&_var_956);
        *(int8_t **)&_var_1614 = fmt_u((unreserved_uintmax_t)((int64_t)((((generic8_t)((int32_t)((generic32_t *)&_var_1346)[22ul] > 4294967295) ? 0ul : 4294967295ul) ^ (generic64_t)(uint64_t)(uint32_t)((generic32_t *)&_var_1346)[22ul]) - ((generic8_t)((int32_t)((generic32_t *)&_var_1346)[22ul] > 4294967295) ? 0ul : 4294967295ul) << 32ul) >> 32l), (int8_t *)&_var_1346 + 107ul);
        _var_1383 = *(generic64_t *)&_var_1614;
        helper_fldt_ST0_wrapper((void *)0ul, (generic64_t)&_var_1346 + 32ul, _var_948, (void *)&_var_923, (void *)&_var_924, (void *)&_var_925, (void *)&_var_926, (void *)&_var_927, (void *)&_var_928, (void *)&_var_929, (void *)&_var_930, (void *)&_var_931, (void *)&_var_932, (void *)&_var_933, (void *)&_var_934, (void *)&_var_935, (void *)&_var_936, (void *)&_var_937, (void *)&_var_938, (void *)&_var_939, (void *)&_var_940, (void *)&_var_941, (void *)&_var_942, (void *)&_var_943, (void *)&_var_944, (void *)&_var_945, (void *)&_var_946, (void *)&_var_947);
        helper_fpush_wrapper((void *)0ul, _var_923, (void *)&_var_914, (void *)&_var_915, (void *)&_var_916, (void *)&_var_917, (void *)&_var_918, (void *)&_var_919, (void *)&_var_920, (void *)&_var_921, (void *)&_var_922);
        helper_fldz_ST0_wrapper((void *)0ul, _var_914, (void *)&_var_898, (void *)&_var_899, (void *)&_var_900, (void *)&_var_901, (void *)&_var_902, (void *)&_var_903, (void *)&_var_904, (void *)&_var_905, (void *)&_var_906, (void *)&_var_907, (void *)&_var_908, (void *)&_var_909, (void *)&_var_910, (void *)&_var_911, (void *)&_var_912, (void *)&_var_913);
        helper_fldt_ST0_wrapper((void *)0ul, 4216032ul, _var_914, (void *)&_var_873, (void *)&_var_874, (void *)&_var_875, (void *)&_var_876, (void *)&_var_877, (void *)&_var_878, (void *)&_var_879, (void *)&_var_880, (void *)&_var_881, (void *)&_var_882, (void *)&_var_883, (void *)&_var_884, (void *)&_var_885, (void *)&_var_886, (void *)&_var_887, (void *)&_var_888, (void *)&_var_889, (void *)&_var_890, (void *)&_var_891, (void *)&_var_892, (void *)&_var_893, (void *)&_var_894, (void *)&_var_895, (void *)&_var_896, (void *)&_var_897);
        *(generic32_t *)&_var_1845 = _var_873;
        *(generic64_t *)&_var_1846 = _var_882;
        *(generic16_t *)&_var_1847 = _var_883;
        *(generic64_t *)&_var_1848 = _var_884;
        *(generic16_t *)&_var_1849 = _var_885;
        *(generic64_t *)&_var_1850 = _var_886;
        *(generic16_t *)&_var_1851 = _var_887;
        *(generic64_t *)&_var_1852 = _var_888;
        *(generic16_t *)&_var_1853 = _var_889;
        *(generic64_t *)&_var_1854 = _var_890;
        *(generic16_t *)&_var_1855 = _var_891;
        *(generic64_t *)&_var_1856 = _var_892;
        *(generic16_t *)&_var_1857 = _var_893;
        *(generic64_t *)&_var_1858 = _var_894;
        *(generic16_t *)&_var_1859 = _var_895;
        *(generic64_t *)&_var_1860 = _var_896;
        *(generic16_t *)&_var_1861 = _var_897;
        if (_var_1383 == (generic64_t)&_var_1346 + 107ul) {
          _var_1383 = (generic64_t)&_var_1346 + 106ul;
          *(generic8_t *)_var_1383 = 48u;
        } else {
        }
        helper_fxchg_ST0_STN_wrapper((void *)0ul, 2u, *(generic32_t *)&_var_1845, *(generic64_t *)&_var_1846, *(generic16_t *)&_var_1847, *(generic64_t *)&_var_1848, *(generic16_t *)&_var_1849, *(generic64_t *)&_var_1850, *(generic16_t *)&_var_1851, *(generic64_t *)&_var_1852, *(generic16_t *)&_var_1853, *(generic64_t *)&_var_1854, *(generic16_t *)&_var_1855, *(generic64_t *)&_var_1856, *(generic16_t *)&_var_1857, *(generic64_t *)&_var_1858, *(generic16_t *)&_var_1859, *(generic64_t *)&_var_1860, *(generic16_t *)&_var_1861, (void *)&_var_813, (void *)&_var_814, (void *)&_var_815, (void *)&_var_816, (void *)&_var_817, (void *)&_var_818, (void *)&_var_819, (void *)&_var_820, (void *)&_var_821, (void *)&_var_822, (void *)&_var_823, (void *)&_var_824, (void *)&_var_825, (void *)&_var_826, (void *)&_var_827, (void *)&_var_828);
        _var_1366 = _var_813;
        _var_1367 = _var_814;
        _var_1368 = _var_815;
        _var_1369 = _var_816;
        _var_1370 = _var_817;
        _var_1371 = _var_818;
        _var_1372 = _var_819;
        _var_1373 = _var_820;
        _var_1374 = _var_821;
        _var_1375 = _var_822;
        _var_1376 = _var_823;
        _var_1377 = _var_824;
        _var_1378 = _var_825;
        _var_1379 = _var_826;
        _var_1380 = _var_827;
        _var_1381 = _var_828;
        *(generic32_t *)&_var_1863 = ((generic32_t *)&_var_1346)[22ul];
        _var_1864 = ((generic8_t *)&_var_1346)[24ul];
        _var_1362 = *(generic64_t *)&_var_1709 & 18446744073709551360ul | (generic64_t)(uint64_t)(uint8_t)_var_1864;
        *(generic64_t *)&_var_1862 = _var_1383 + 18446744073709551614ul;
        *(generic32_t *)&_var_1865 = ((generic32_t *)&_var_1346)[5ul] & 8u;
        ((generic8_t *)_var_1383)[18446744073709551615ul] = ((generic8_t)(generic32_t)((uint32_t)*(generic32_t *)&_var_1863 >> 30u) & 2u) + 43u;
        *(generic8_t *)*(generic64_t *)&_var_1862 = _var_1864 + 15u;
        *(generic32_t *)&_var_1615 = helper_fnstcw_wrapper((void *)0ul, 895u);
        ((generic16_t *)&_var_1346)[39ul] = (generic16_t)*(generic32_t *)&_var_1615;
        ((generic16_t *)&_var_1346)[38ul] = (generic16_t)*(generic32_t *)&_var_1615 | 3072u;
        _var_1363 = (generic64_t)&_var_1346 + 107ul;
        _var_1364 = *(generic32_t *)&_var_1845;
        _var_1365 = 895u;
        _var_1382 = _var_1844;
      _label_0:
        *(generic64_t *)&_var_1866 = _var_1362;
        *(generic64_t *)&_var_1867 = _var_1363;
        helper_fpush_wrapper((void *)0ul, _var_1364, (void *)&_var_688, (void *)&_var_689, (void *)&_var_690, (void *)&_var_691, (void *)&_var_692, (void *)&_var_693, (void *)&_var_694, (void *)&_var_695, (void *)&_var_696);
        *(generic32_t *)&_var_1868 = _var_688;
        helper_fmov_ST0_STN_wrapper((void *)0ul, 1u, *(generic32_t *)&_var_1868, _var_1366, _var_1367, _var_1368, _var_1369, _var_1370, _var_1371, _var_1372, _var_1373, _var_1374, _var_1375, _var_1376, _var_1377, _var_1378, _var_1379, _var_1380, _var_1381, (void *)&_var_672, (void *)&_var_673, (void *)&_var_674, (void *)&_var_675, (void *)&_var_676, (void *)&_var_677, (void *)&_var_678, (void *)&_var_679, (void *)&_var_680, (void *)&_var_681, (void *)&_var_682, (void *)&_var_683, (void *)&_var_684, (void *)&_var_685, (void *)&_var_686, (void *)&_var_687);
        *(generic64_t *)&_var_1869 = _var_672;
        *(generic16_t *)&_var_1870 = _var_673;
        *(generic64_t *)&_var_1871 = _var_674;
        *(generic16_t *)&_var_1872 = _var_675;
        *(generic64_t *)&_var_1873 = _var_676;
        *(generic16_t *)&_var_1874 = _var_677;
        *(generic64_t *)&_var_1875 = _var_678;
        *(generic16_t *)&_var_1876 = _var_679;
        *(generic64_t *)&_var_1877 = _var_680;
        *(generic16_t *)&_var_1878 = _var_681;
        *(generic64_t *)&_var_1879 = _var_682;
        *(generic16_t *)&_var_1880 = _var_683;
        *(generic64_t *)&_var_1881 = _var_684;
        *(generic16_t *)&_var_1882 = _var_685;
        *(generic64_t *)&_var_1883 = _var_686;
        *(generic16_t *)&_var_1884 = _var_687;
        helper_fldcw_wrapper((void *)0ul, (generic32_t)(uint32_t)(uint16_t)((generic16_t *)&_var_1346)[38ul], _var_1365, (void *)&_var_669, (void *)&_var_670, (void *)&_var_671);
        *(generic16_t *)&_var_1885 = _var_669;
        *(generic32_t *)&_var_1616 = helper_fistl_ST0_wrapper((void *)0ul, *(generic32_t *)&_var_1868, *(generic64_t *)&_var_1869, *(generic16_t *)&_var_1870, *(generic64_t *)&_var_1871, *(generic16_t *)&_var_1872, *(generic64_t *)&_var_1873, *(generic16_t *)&_var_1874, *(generic64_t *)&_var_1875, *(generic16_t *)&_var_1876, *(generic64_t *)&_var_1877, *(generic16_t *)&_var_1878, *(generic64_t *)&_var_1879, *(generic16_t *)&_var_1880, *(generic64_t *)&_var_1881, *(generic16_t *)&_var_1882, *(generic64_t *)&_var_1883, *(generic16_t *)&_var_1884, _var_670, _var_1382, (void *)&_var_668);
        _var_1886 = _var_668;
        ((generic32_t *)&_var_1346)[6ul] = *(generic32_t *)&_var_1616;
        helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1868, (void *)&_var_659, (void *)&_var_660, (void *)&_var_661, (void *)&_var_662, (void *)&_var_663, (void *)&_var_664, (void *)&_var_665, (void *)&_var_666, (void *)&_var_667);
        *(generic32_t *)&_var_1887 = _var_659;
        helper_fldcw_wrapper((void *)0ul, (generic32_t)(uint32_t)(uint16_t)((generic16_t *)&_var_1346)[39ul], *(generic16_t *)&_var_1885, (void *)&_var_656, (void *)&_var_657, (void *)&_var_658);
        *(generic16_t *)&_var_1888 = _var_656;
        helper_fildl_FT0_wrapper((void *)0ul, ((generic32_t *)&_var_1346)[6ul], (void *)&_var_654, (void *)&_var_655);
        helper_fsub_ST0_FT0_wrapper((void *)0ul, *(generic32_t *)&_var_1887, *(generic64_t *)&_var_1869, *(generic16_t *)&_var_1870, *(generic64_t *)&_var_1871, *(generic16_t *)&_var_1872, *(generic64_t *)&_var_1873, *(generic16_t *)&_var_1874, *(generic64_t *)&_var_1875, *(generic16_t *)&_var_1876, *(generic64_t *)&_var_1877, *(generic16_t *)&_var_1878, *(generic64_t *)&_var_1879, *(generic16_t *)&_var_1880, *(generic64_t *)&_var_1881, *(generic16_t *)&_var_1882, *(generic64_t *)&_var_1883, *(generic16_t *)&_var_1884, 0u, _var_657, _var_1886, _var_658, 0u, 0u, _var_654, _var_655, (void *)&_var_637, (void *)&_var_638, (void *)&_var_639, (void *)&_var_640, (void *)&_var_641, (void *)&_var_642, (void *)&_var_643, (void *)&_var_644, (void *)&_var_645, (void *)&_var_646, (void *)&_var_647, (void *)&_var_648, (void *)&_var_649, (void *)&_var_650, (void *)&_var_651, (void *)&_var_652, (void *)&_var_653);
        _var_1889 = ((generic8_t *)(int64_t)(int32_t)((generic32_t *)&_var_1346)[6ul])[4215424ul];
        _var_1353 = *(generic64_t *)&_var_1867 + 1ul;
        _var_1354 = *(generic64_t *)&_var_1866 & 4294967040ul | (generic64_t)(uint64_t)(uint8_t)(_var_1864 & 32u | _var_1889);
        helper_fmov_FT0_STN_wrapper((void *)0ul, 2u, *(generic32_t *)&_var_1887, _var_637, _var_638, _var_639, _var_640, _var_641, _var_642, _var_643, _var_644, _var_645, _var_646, _var_647, _var_648, _var_649, _var_650, _var_651, _var_652, (void *)&_var_635, (void *)&_var_636);
        helper_fmul_ST0_FT0_wrapper((void *)0ul, *(generic32_t *)&_var_1887, _var_637, _var_638, _var_639, _var_640, _var_641, _var_642, _var_643, _var_644, _var_645, _var_646, _var_647, _var_648, _var_649, _var_650, _var_651, _var_652, 0u, _var_657, _var_653, _var_658, 0u, 0u, _var_635, _var_636, (void *)&_var_618, (void *)&_var_619, (void *)&_var_620, (void *)&_var_621, (void *)&_var_622, (void *)&_var_623, (void *)&_var_624, (void *)&_var_625, (void *)&_var_626, (void *)&_var_627, (void *)&_var_628, (void *)&_var_629, (void *)&_var_630, (void *)&_var_631, (void *)&_var_632, (void *)&_var_633, (void *)&_var_634);
        *(generic64_t *)&_var_1890 = _var_618;
        *(generic16_t *)&_var_1891 = _var_619;
        *(generic64_t *)&_var_1892 = _var_620;
        *(generic16_t *)&_var_1893 = _var_621;
        *(generic64_t *)&_var_1894 = _var_622;
        *(generic16_t *)&_var_1895 = _var_623;
        *(generic64_t *)&_var_1896 = _var_624;
        *(generic16_t *)&_var_1897 = _var_625;
        *(generic64_t *)&_var_1898 = _var_626;
        *(generic16_t *)&_var_1899 = _var_627;
        *(generic64_t *)&_var_1900 = _var_628;
        *(generic16_t *)&_var_1901 = _var_629;
        *(generic64_t *)&_var_1902 = _var_630;
        *(generic16_t *)&_var_1903 = _var_631;
        *(generic64_t *)&_var_1904 = _var_632;
        *(generic16_t *)&_var_1905 = _var_633;
        _var_1906 = _var_634;
        _var_1355 = _var_1906;
        *(generic8_t *)*(generic64_t *)&_var_1867 = _var_1864 & 32u | _var_1889;
        _var_1358 = *(generic64_t *)&_var_1867 - ((generic64_t)&_var_1346 + 107ul);
        _var_1356 = (generic64_t)&_var_1346 + 107ul;
        _var_1357 = 17u;
        if (*(generic64_t *)&_var_1867 == (generic64_t)&_var_1346 + 107ul) {
          helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, *(generic32_t *)&_var_1887, *(generic64_t *)&_var_1890, *(generic16_t *)&_var_1891, *(generic64_t *)&_var_1892, *(generic16_t *)&_var_1893, *(generic64_t *)&_var_1894, *(generic16_t *)&_var_1895, *(generic64_t *)&_var_1896, *(generic16_t *)&_var_1897, *(generic64_t *)&_var_1898, *(generic16_t *)&_var_1899, *(generic64_t *)&_var_1900, *(generic16_t *)&_var_1901, *(generic64_t *)&_var_1902, *(generic16_t *)&_var_1903, *(generic64_t *)&_var_1904, *(generic16_t *)&_var_1905, (void *)&_var_480, (void *)&_var_481);
          helper_fucomi_ST0_FT0_wrapper((void *)0ul, *(generic64_t *)&_var_1867 - ((generic64_t)&_var_1346 + 107ul), 17u, (generic64_t)&_var_1346 + 107ul, 0ul, *(generic32_t *)&_var_1887, *(generic64_t *)&_var_1890, *(generic16_t *)&_var_1891, *(generic64_t *)&_var_1892, *(generic16_t *)&_var_1893, *(generic64_t *)&_var_1894, *(generic16_t *)&_var_1895, *(generic64_t *)&_var_1896, *(generic16_t *)&_var_1897, *(generic64_t *)&_var_1898, *(generic16_t *)&_var_1899, *(generic64_t *)&_var_1900, *(generic16_t *)&_var_1901, *(generic64_t *)&_var_1902, *(generic16_t *)&_var_1903, *(generic64_t *)&_var_1904, *(generic16_t *)&_var_1905, _var_1906, _var_480, _var_481, (void *)&_var_478, (void *)&_var_479);
          _var_1359 = _var_478;
          _var_1907 = _var_479;
          _var_1361 = ((_var_1359 & 64ul) == 0ul ? (generic64_t)&_var_1346 + 107ul & 18446744073709551360ul | 1ul : *(generic64_t *)&_var_1866 & 4294967040ul | (generic64_t)((uint64_t)_var_1359 >> 2ul) & 1ul) & 4294967041ul;
          _var_1360 = 22u;
          if ((((_var_1359 & 64ul) == 0ul ? (generic64_t)&_var_1346 + 107ul & 18446744073709551360ul | 1ul : *(generic64_t *)&_var_1866 & 4294967040ul | (generic64_t)((uint64_t)_var_1359 >> 2ul) & 1ul) & 1ul) == 0ul) {
            *(generic64_t *)&_var_1617 = lshift((generic64_t)(uint64_t)(uint32_t)p, 4294967272u);
            _var_1359 = (generic64_t)(uint64_t)(uint32_t)((llvm.ctpop.i32((generic32_t)p & 255u) << 2u & 4u | ((generic32_t)p == 0u ? 64u : 0u) | (generic32_t)*(generic64_t *)&_var_1617 & 128u) ^ 4u);
            _var_1360 = (generic8_t)((uint8_t)((generic8_t)(generic64_t)((uint64_t)_var_1359 >> 4ul) ^ (generic8_t)(llvm.ctpop.i32((generic32_t)p & 255u) << 2u & 4u | ((generic32_t)p == 0u ? 64u : 0u) | (generic32_t)*(generic64_t *)&_var_1617 & 128u)) < 64u) ? 1u : 24u;
            _var_1361 = (generic8_t)((uint8_t)((generic8_t)(generic64_t)((uint64_t)_var_1359 >> 4ul) ^ (generic8_t)(llvm.ctpop.i32((generic32_t)p & 255u) << 2u & 4u | ((generic32_t)p == 0u ? 64u : 0u) | (generic32_t)*(generic64_t *)&_var_1617 & 128u)) < 64u) ? (generic64_t)(uint64_t)(uint32_t)p : (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1865;
            if ((generic8_t)((uint8_t)((generic8_t)(generic64_t)((uint64_t)_var_1359 >> 4ul) ^ (generic8_t)(llvm.ctpop.i32((generic32_t)p & 255u) << 2u & 4u | ((generic32_t)p == 0u ? 64u : 0u) | (generic32_t)*(generic64_t *)&_var_1617 & 128u)) < 64u) ? 1u : (generic8_t)(*(generic32_t *)&_var_1865 != 0u)) {
              _var_1356 = _var_1359;
              _var_1357 = _var_1360;
              _var_1358 = _var_1361;
              ((generic8_t *)*(generic64_t *)&_var_1867)[1ul] = 46u;
              _var_1353 = (generic64_t)&_var_1346 + 109ul;
              _var_1354 = _var_1361;
              _var_1355 = _var_1907;
              helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, *(generic32_t *)&_var_1887, *(generic64_t *)&_var_1890, *(generic16_t *)&_var_1891, *(generic64_t *)&_var_1892, *(generic16_t *)&_var_1893, *(generic64_t *)&_var_1894, *(generic16_t *)&_var_1895, *(generic64_t *)&_var_1896, *(generic16_t *)&_var_1897, *(generic64_t *)&_var_1898, *(generic16_t *)&_var_1899, *(generic64_t *)&_var_1900, *(generic16_t *)&_var_1901, *(generic64_t *)&_var_1902, *(generic16_t *)&_var_1903, *(generic64_t *)&_var_1904, *(generic16_t *)&_var_1905, (void *)&_var_476, (void *)&_var_477);
              helper_fucomi_ST0_FT0_wrapper((void *)0ul, _var_1358, _var_1357, _var_1356, 0ul, *(generic32_t *)&_var_1887, *(generic64_t *)&_var_1890, *(generic16_t *)&_var_1891, *(generic64_t *)&_var_1892, *(generic16_t *)&_var_1893, *(generic64_t *)&_var_1894, *(generic16_t *)&_var_1895, *(generic64_t *)&_var_1896, *(generic16_t *)&_var_1897, *(generic64_t *)&_var_1898, *(generic16_t *)&_var_1899, *(generic64_t *)&_var_1900, *(generic16_t *)&_var_1901, *(generic64_t *)&_var_1902, *(generic16_t *)&_var_1903, *(generic64_t *)&_var_1904, *(generic16_t *)&_var_1905, _var_1355, _var_476, _var_477, (void *)&_var_474, (void *)&_var_475);
              _var_1382 = _var_475;
              _var_1364 = *(generic32_t *)&_var_1887;
              _var_1365 = *(generic16_t *)&_var_1888;
              _var_1366 = *(generic64_t *)&_var_1890;
              _var_1367 = *(generic16_t *)&_var_1891;
              _var_1368 = *(generic64_t *)&_var_1892;
              _var_1369 = *(generic16_t *)&_var_1893;
              _var_1370 = *(generic64_t *)&_var_1894;
              _var_1371 = *(generic16_t *)&_var_1895;
              _var_1372 = *(generic64_t *)&_var_1896;
              _var_1373 = *(generic16_t *)&_var_1897;
              _var_1374 = *(generic64_t *)&_var_1898;
              _var_1375 = *(generic16_t *)&_var_1899;
              _var_1376 = *(generic64_t *)&_var_1900;
              _var_1377 = *(generic16_t *)&_var_1901;
              _var_1378 = *(generic64_t *)&_var_1902;
              _var_1379 = *(generic16_t *)&_var_1903;
              _var_1380 = *(generic64_t *)&_var_1904;
              _var_1381 = *(generic16_t *)&_var_1905;
              if ((_var_474 & 68ul) == 64ul) {
                helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, *(generic32_t *)&_var_1887, *(generic64_t *)&_var_1890, *(generic16_t *)&_var_1891, *(generic64_t *)&_var_1892, *(generic16_t *)&_var_1893, *(generic64_t *)&_var_1894, *(generic16_t *)&_var_1895, *(generic64_t *)&_var_1896, *(generic16_t *)&_var_1897, *(generic64_t *)&_var_1898, *(generic16_t *)&_var_1899, *(generic64_t *)&_var_1900, *(generic16_t *)&_var_1901, *(generic64_t *)&_var_1902, *(generic16_t *)&_var_1903, *(generic64_t *)&_var_1904, *(generic16_t *)&_var_1905, (void *)&_var_408, (void *)&_var_409, (void *)&_var_410, (void *)&_var_411, (void *)&_var_412, (void *)&_var_413, (void *)&_var_414, (void *)&_var_415, (void *)&_var_416, (void *)&_var_417, (void *)&_var_418, (void *)&_var_419, (void *)&_var_420, (void *)&_var_421, (void *)&_var_422, (void *)&_var_423);
                helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1887, (void *)&_var_399, (void *)&_var_400, (void *)&_var_401, (void *)&_var_402, (void *)&_var_403, (void *)&_var_404, (void *)&_var_405, (void *)&_var_406, (void *)&_var_407);
                helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_399, _var_408, _var_409, _var_410, _var_411, _var_412, _var_413, _var_414, _var_415, _var_416, _var_417, _var_418, _var_419, _var_420, _var_421, _var_422, _var_423, (void *)&_var_383, (void *)&_var_384, (void *)&_var_385, (void *)&_var_386, (void *)&_var_387, (void *)&_var_388, (void *)&_var_389, (void *)&_var_390, (void *)&_var_391, (void *)&_var_392, (void *)&_var_393, (void *)&_var_394, (void *)&_var_395, (void *)&_var_396, (void *)&_var_397, (void *)&_var_398);
                helper_fpop_wrapper((void *)0ul, _var_399, (void *)&_var_374, (void *)&_var_375, (void *)&_var_376, (void *)&_var_377, (void *)&_var_378, (void *)&_var_379, (void *)&_var_380, (void *)&_var_381, (void *)&_var_382);
                helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_374, _var_383, _var_384, _var_385, _var_386, _var_387, _var_388, _var_389, _var_390, _var_391, _var_392, _var_393, _var_394, _var_395, _var_396, _var_397, _var_398, (void *)&_var_358, (void *)&_var_359, (void *)&_var_360, (void *)&_var_361, (void *)&_var_362, (void *)&_var_363, (void *)&_var_364, (void *)&_var_365, (void *)&_var_366, (void *)&_var_367, (void *)&_var_368, (void *)&_var_369, (void *)&_var_370, (void *)&_var_371, (void *)&_var_372, (void *)&_var_373);
                helper_fpop_wrapper((void *)0ul, _var_374, (void *)&_var_349, (void *)&_var_350, (void *)&_var_351, (void *)&_var_352, (void *)&_var_353, (void *)&_var_354, (void *)&_var_355, (void *)&_var_356, (void *)&_var_357);
                _var_1352 = _var_1353;
                *(generic64_t *)&_var_1908 = _var_1352 - ((generic64_t)&_var_1346 + 107ul);
                if ((generic32_t)p == 0u) {
                  _var_1351 = *(generic64_t *)&_var_1908 + ((generic64_t)&_var_1346 + 107ul - *(generic64_t *)&_var_1862);
                } else {
                  _var_1351 = (generic64_t)&_var_1346 + 107ul - *(generic64_t *)&_var_1862 + (generic64_t)(uint64_t)(uint32_t)p + 2ul;
                  if ((generic8_t)((int64_t)(*(generic64_t *)&_var_1908 + 18446744073709551615ul) > (int64_t)p)) {
                  } else {
                  }
                }
                *(generic64_t *)&_var_1909 = _var_1351;
                *(generic32_t *)&_var_1910 = ((generic32_t *)&_var_1346)[12ul];
                *(generic32_t *)&_var_1911 = ((generic32_t *)&_var_1346)[4ul];
                *(generic32_t *)&_var_1912 = ((generic32_t *)&_var_1346)[5ul];
                ((generic64_t *)&_var_1346)[3ul] = *(generic64_t *)&_var_1862;
                _var_1348 = *(generic32_t *)&_var_1910 + (generic32_t)*(generic64_t *)&_var_1909;
                pad(f, 32, (int32_t)*(generic32_t *)&_var_1911, (int32_t)_var_1348, (int32_t)*(generic32_t *)&_var_1912);
                out(f, (const int8_t *)*(generic64_t *)&_var_1808, (size_t)(int64_t)(int32_t)((generic32_t *)&_var_1346)[12ul]);
                pad(f, 48, (int32_t)((generic32_t *)&_var_1346)[4ul], (int32_t)_var_1348, (int32_t)(((generic32_t *)&_var_1346)[5ul] ^ 65536u));
                out(f, (const int8_t *)&_var_1346 + 107ul, (size_t)*(generic64_t *)&_var_1908);
                pad(f, 48, (int32_t)(generic32_t)(*(generic64_t *)&_var_1909 - (*(generic64_t *)&_var_1908 + ((generic64_t)&_var_1346 + 107ul - *(generic64_t *)&_var_1862))), 0, 0);
                _var_1350 = ((generic64_t *)&_var_1346)[3ul];
                _var_1349 = (generic64_t)&_var_1346 + 107ul - *(generic64_t *)&_var_1862;
                out(f, (const int8_t *)_var_1350, (size_t)_var_1349);
                pad(f, 32, (int32_t)((generic32_t *)&_var_1346)[4ul], (int32_t)_var_1348, (int32_t)(((generic32_t *)&_var_1346)[5ul] ^ 8192u));
                _var_1347 = (generic8_t)((int64_t)((generic64_t)(uint64_t)(uint32_t)_var_1348 << 32ul) < (int64_t)((generic64_t)(uint64_t)(uint32_t)((generic32_t *)&_var_1346)[4ul] << 32ul)) ? ((generic32_t *)&_var_1346)[4ul] : _var_1348;
              } else {
                goto _label_0;
              }
            } else {
              helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, *(generic32_t *)&_var_1887, *(generic64_t *)&_var_1890, *(generic16_t *)&_var_1891, *(generic64_t *)&_var_1892, *(generic16_t *)&_var_1893, *(generic64_t *)&_var_1894, *(generic16_t *)&_var_1895, *(generic64_t *)&_var_1896, *(generic16_t *)&_var_1897, *(generic64_t *)&_var_1898, *(generic16_t *)&_var_1899, *(generic64_t *)&_var_1900, *(generic16_t *)&_var_1901, *(generic64_t *)&_var_1902, *(generic16_t *)&_var_1903, *(generic64_t *)&_var_1904, *(generic16_t *)&_var_1905, (void *)&_var_333, (void *)&_var_334, (void *)&_var_335, (void *)&_var_336, (void *)&_var_337, (void *)&_var_338, (void *)&_var_339, (void *)&_var_340, (void *)&_var_341, (void *)&_var_342, (void *)&_var_343, (void *)&_var_344, (void *)&_var_345, (void *)&_var_346, (void *)&_var_347, (void *)&_var_348);
              helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1887, (void *)&_var_324, (void *)&_var_325, (void *)&_var_326, (void *)&_var_327, (void *)&_var_328, (void *)&_var_329, (void *)&_var_330, (void *)&_var_331, (void *)&_var_332);
              helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_324, _var_333, _var_334, _var_335, _var_336, _var_337, _var_338, _var_339, _var_340, _var_341, _var_342, _var_343, _var_344, _var_345, _var_346, _var_347, _var_348, (void *)&_var_308, (void *)&_var_309, (void *)&_var_310, (void *)&_var_311, (void *)&_var_312, (void *)&_var_313, (void *)&_var_314, (void *)&_var_315, (void *)&_var_316, (void *)&_var_317, (void *)&_var_318, (void *)&_var_319, (void *)&_var_320, (void *)&_var_321, (void *)&_var_322, (void *)&_var_323);
              helper_fpop_wrapper((void *)0ul, _var_324, (void *)&_var_299, (void *)&_var_300, (void *)&_var_301, (void *)&_var_302, (void *)&_var_303, (void *)&_var_304, (void *)&_var_305, (void *)&_var_306, (void *)&_var_307);
              helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_299, _var_308, _var_309, _var_310, _var_311, _var_312, _var_313, _var_314, _var_315, _var_316, _var_317, _var_318, _var_319, _var_320, _var_321, _var_322, _var_323, (void *)&_var_283, (void *)&_var_284, (void *)&_var_285, (void *)&_var_286, (void *)&_var_287, (void *)&_var_288, (void *)&_var_289, (void *)&_var_290, (void *)&_var_291, (void *)&_var_292, (void *)&_var_293, (void *)&_var_294, (void *)&_var_295, (void *)&_var_296, (void *)&_var_297, (void *)&_var_298);
              helper_fpop_wrapper((void *)0ul, _var_299, (void *)&_var_274, (void *)&_var_275, (void *)&_var_276, (void *)&_var_277, (void *)&_var_278, (void *)&_var_279, (void *)&_var_280, (void *)&_var_281, (void *)&_var_282);
              _var_1352 = *(generic64_t *)&_var_1867 + 1ul;
            }
          } else {
          }
        } else {
        }
      } else {
        helper_flds_ST0_wrapper((void *)0ul, *(generic32_t *)4215976ul, *(generic32_t *)&_var_1809, _var_1826, 0u, 0u, (void *)&_var_977, (void *)&_var_978, (void *)&_var_979, (void *)&_var_980, (void *)&_var_981, (void *)&_var_982, (void *)&_var_983, (void *)&_var_984, (void *)&_var_985, (void *)&_var_986, (void *)&_var_987, (void *)&_var_988, (void *)&_var_989, (void *)&_var_990, (void *)&_var_991, (void *)&_var_992, (void *)&_var_993, (void *)&_var_994, (void *)&_var_995, (void *)&_var_996, (void *)&_var_997, (void *)&_var_998, (void *)&_var_999, (void *)&_var_1000, (void *)&_var_1001, (void *)&_var_1002);
        _var_1402 = 15ul - (generic64_t)(uint64_t)(uint32_t)p;
        *(generic64_t **)&_var_1403 = &_var_986;
        *(generic16_t **)&_var_1404 = &_var_987;
        *(generic64_t **)&_var_1405 = &_var_988;
        *(generic16_t **)&_var_1406 = &_var_989;
        *(generic64_t **)&_var_1407 = &_var_990;
        *(generic16_t **)&_var_1408 = &_var_991;
        *(generic64_t **)&_var_1409 = &_var_992;
        *(generic16_t **)&_var_1410 = &_var_993;
        *(generic64_t **)&_var_1411 = &_var_994;
        *(generic16_t **)&_var_1412 = &_var_995;
        *(generic64_t **)&_var_1413 = &_var_996;
        *(generic16_t **)&_var_1414 = &_var_997;
        *(generic64_t **)&_var_1415 = &_var_998;
        *(generic16_t **)&_var_1416 = &_var_999;
        *(generic64_t **)&_var_1417 = &_var_1000;
        *(generic16_t **)&_var_1418 = &_var_1001;
        *(generic8_t **)&_var_1419 = &_var_1002;
      _label_1:
        _var_1827 = *(generic8_t *)_var_1419;
        *(generic16_t *)&_var_1828 = *(generic16_t *)_var_1418;
        *(generic64_t *)&_var_1829 = *(generic64_t *)_var_1417;
        *(generic16_t *)&_var_1830 = *(generic16_t *)_var_1416;
        *(generic64_t *)&_var_1831 = *(generic64_t *)_var_1415;
        *(generic16_t *)&_var_1832 = *(generic16_t *)_var_1414;
        *(generic64_t *)&_var_1833 = *(generic64_t *)_var_1413;
        *(generic16_t *)&_var_1834 = *(generic16_t *)_var_1412;
        *(generic64_t *)&_var_1835 = *(generic64_t *)_var_1411;
        *(generic16_t *)&_var_1836 = *(generic16_t *)_var_1410;
        *(generic64_t *)&_var_1837 = *(generic64_t *)_var_1409;
        *(generic16_t *)&_var_1838 = *(generic16_t *)_var_1408;
        *(generic64_t *)&_var_1839 = *(generic64_t *)_var_1407;
        *(generic16_t *)&_var_1840 = *(generic16_t *)_var_1406;
        *(generic64_t *)&_var_1841 = *(generic64_t *)_var_1405;
        *(generic16_t *)&_var_1842 = *(generic16_t *)_var_1404;
        *(generic64_t *)&_var_1843 = *(generic64_t *)_var_1403;
        if ((_var_1402 & 4294967295ul) == 0ul) {
          helper_fmov_STN_ST0_wrapper((void *)0ul, 1u, _var_977, *(generic64_t *)&_var_1843, *(generic16_t *)&_var_1842, *(generic64_t *)&_var_1841, *(generic16_t *)&_var_1840, *(generic64_t *)&_var_1839, *(generic16_t *)&_var_1838, *(generic64_t *)&_var_1837, *(generic16_t *)&_var_1836, *(generic64_t *)&_var_1835, *(generic16_t *)&_var_1834, *(generic64_t *)&_var_1833, *(generic16_t *)&_var_1832, *(generic64_t *)&_var_1831, *(generic16_t *)&_var_1830, *(generic64_t *)&_var_1829, *(generic16_t *)&_var_1828, (void *)&_var_857, (void *)&_var_858, (void *)&_var_859, (void *)&_var_860, (void *)&_var_861, (void *)&_var_862, (void *)&_var_863, (void *)&_var_864, (void *)&_var_865, (void *)&_var_866, (void *)&_var_867, (void *)&_var_868, (void *)&_var_869, (void *)&_var_870, (void *)&_var_871, (void *)&_var_872);
          helper_fpop_wrapper((void *)0ul, _var_977, (void *)&_var_848, (void *)&_var_849, (void *)&_var_850, (void *)&_var_851, (void *)&_var_852, (void *)&_var_853, (void *)&_var_854, (void *)&_var_855, (void *)&_var_856);
          if (*(generic8_t *)*(generic64_t *)&_var_1808 == 45u) {
            helper_fxchg_ST0_STN_wrapper((void *)0ul, 1u, _var_848, _var_857, _var_858, _var_859, _var_860, _var_861, _var_862, _var_863, _var_864, _var_865, _var_866, _var_867, _var_868, _var_869, _var_870, _var_871, _var_872, (void *)&_var_602, (void *)&_var_603, (void *)&_var_604, (void *)&_var_605, (void *)&_var_606, (void *)&_var_607, (void *)&_var_608, (void *)&_var_609, (void *)&_var_610, (void *)&_var_611, (void *)&_var_612, (void *)&_var_613, (void *)&_var_614, (void *)&_var_615, (void *)&_var_616, (void *)&_var_617);
            helper_fchs_ST0_wrapper((void *)0ul, _var_848, _var_602, _var_603, _var_604, _var_605, _var_606, _var_607, _var_608, _var_609, _var_610, _var_611, _var_612, _var_613, _var_614, _var_615, _var_616, _var_617, (void *)&_var_586, (void *)&_var_587, (void *)&_var_588, (void *)&_var_589, (void *)&_var_590, (void *)&_var_591, (void *)&_var_592, (void *)&_var_593, (void *)&_var_594, (void *)&_var_595, (void *)&_var_596, (void *)&_var_597, (void *)&_var_598, (void *)&_var_599, (void *)&_var_600, (void *)&_var_601);
            helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, _var_848, _var_586, _var_587, _var_588, _var_589, _var_590, _var_591, _var_592, _var_593, _var_594, _var_595, _var_596, _var_597, _var_598, _var_599, _var_600, _var_601, (void *)&_var_584, (void *)&_var_585);
            helper_fsub_ST0_FT0_wrapper((void *)0ul, _var_848, _var_586, _var_587, _var_588, _var_589, _var_590, _var_591, _var_592, _var_593, _var_594, _var_595, _var_596, _var_597, _var_598, _var_599, _var_600, _var_601, 0u, 0u, _var_1827, 80u, 0u, 0u, _var_584, _var_585, (void *)&_var_567, (void *)&_var_568, (void *)&_var_569, (void *)&_var_570, (void *)&_var_571, (void *)&_var_572, (void *)&_var_573, (void *)&_var_574, (void *)&_var_575, (void *)&_var_576, (void *)&_var_577, (void *)&_var_578, (void *)&_var_579, (void *)&_var_580, (void *)&_var_581, (void *)&_var_582, (void *)&_var_583);
            helper_fadd_STN_ST0_wrapper((void *)0ul, 1u, _var_848, _var_567, _var_568, _var_569, _var_570, _var_571, _var_572, _var_573, _var_574, _var_575, _var_576, _var_577, _var_578, _var_579, _var_580, _var_581, _var_582, 0u, 0u, _var_583, 80u, 0u, 0u, (void *)&_var_550, (void *)&_var_551, (void *)&_var_552, (void *)&_var_553, (void *)&_var_554, (void *)&_var_555, (void *)&_var_556, (void *)&_var_557, (void *)&_var_558, (void *)&_var_559, (void *)&_var_560, (void *)&_var_561, (void *)&_var_562, (void *)&_var_563, (void *)&_var_564, (void *)&_var_565, (void *)&_var_566);
            _var_1401 = _var_566;
            helper_fpop_wrapper((void *)0ul, _var_848, (void *)&_var_541, (void *)&_var_542, (void *)&_var_543, (void *)&_var_544, (void *)&_var_545, (void *)&_var_546, (void *)&_var_547, (void *)&_var_548, (void *)&_var_549);
            _var_1384 = _var_541;
            helper_fchs_ST0_wrapper((void *)0ul, _var_1384, _var_550, _var_551, _var_552, _var_553, _var_554, _var_555, _var_556, _var_557, _var_558, _var_559, _var_560, _var_561, _var_562, _var_563, _var_564, _var_565, (void *)&_var_525, (void *)&_var_526, (void *)&_var_527, (void *)&_var_528, (void *)&_var_529, (void *)&_var_530, (void *)&_var_531, (void *)&_var_532, (void *)&_var_533, (void *)&_var_534, (void *)&_var_535, (void *)&_var_536, (void *)&_var_537, (void *)&_var_538, (void *)&_var_539, (void *)&_var_540);
            _var_1385 = _var_525;
            _var_1386 = _var_526;
            _var_1387 = _var_527;
            _var_1388 = _var_528;
            _var_1389 = _var_529;
            _var_1390 = _var_530;
            _var_1391 = _var_531;
            _var_1392 = _var_532;
            _var_1393 = _var_533;
            _var_1394 = _var_534;
            _var_1395 = _var_535;
            _var_1396 = _var_536;
            _var_1397 = _var_537;
            _var_1398 = _var_538;
            _var_1399 = _var_539;
            _var_1400 = _var_540;
          } else {
            helper_fadd_STN_ST0_wrapper((void *)0ul, 1u, _var_848, _var_857, _var_858, _var_859, _var_860, _var_861, _var_862, _var_863, _var_864, _var_865, _var_866, _var_867, _var_868, _var_869, _var_870, _var_871, _var_872, 0u, 0u, _var_1827, 80u, 0u, 0u, (void *)&_var_508, (void *)&_var_509, (void *)&_var_510, (void *)&_var_511, (void *)&_var_512, (void *)&_var_513, (void *)&_var_514, (void *)&_var_515, (void *)&_var_516, (void *)&_var_517, (void *)&_var_518, (void *)&_var_519, (void *)&_var_520, (void *)&_var_521, (void *)&_var_522, (void *)&_var_523, (void *)&_var_524);
            helper_fsub_STN_ST0_wrapper((void *)0ul, 1u, _var_848, _var_508, _var_509, _var_510, _var_511, _var_512, _var_513, _var_514, _var_515, _var_516, _var_517, _var_518, _var_519, _var_520, _var_521, _var_522, _var_523, 0u, 0u, _var_524, 80u, 0u, 0u, (void *)&_var_491, (void *)&_var_492, (void *)&_var_493, (void *)&_var_494, (void *)&_var_495, (void *)&_var_496, (void *)&_var_497, (void *)&_var_498, (void *)&_var_499, (void *)&_var_500, (void *)&_var_501, (void *)&_var_502, (void *)&_var_503, (void *)&_var_504, (void *)&_var_505, (void *)&_var_506, (void *)&_var_507);
            _var_1385 = _var_491;
            _var_1386 = _var_492;
            _var_1387 = _var_493;
            _var_1388 = _var_494;
            _var_1389 = _var_495;
            _var_1390 = _var_496;
            _var_1391 = _var_497;
            _var_1392 = _var_498;
            _var_1393 = _var_499;
            _var_1394 = _var_500;
            _var_1395 = _var_501;
            _var_1396 = _var_502;
            _var_1397 = _var_503;
            _var_1398 = _var_504;
            _var_1399 = _var_505;
            _var_1400 = _var_506;
            _var_1401 = _var_507;
            helper_fpop_wrapper((void *)0ul, _var_848, (void *)&_var_482, (void *)&_var_483, (void *)&_var_484, (void *)&_var_485, (void *)&_var_486, (void *)&_var_487, (void *)&_var_488, (void *)&_var_489, (void *)&_var_490);
            _var_1384 = _var_482;
          }
        } else {
          _var_1402 = (_var_1402 & 4294967295ul) + 4294967295ul;
          helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, _var_977, *(generic64_t *)&_var_1843, *(generic16_t *)&_var_1842, *(generic64_t *)&_var_1841, *(generic16_t *)&_var_1840, *(generic64_t *)&_var_1839, *(generic16_t *)&_var_1838, *(generic64_t *)&_var_1837, *(generic16_t *)&_var_1836, *(generic64_t *)&_var_1835, *(generic16_t *)&_var_1834, *(generic64_t *)&_var_1833, *(generic16_t *)&_var_1832, *(generic64_t *)&_var_1831, *(generic16_t *)&_var_1830, *(generic64_t *)&_var_1829, *(generic16_t *)&_var_1828, (void *)&_var_846, (void *)&_var_847);
          helper_fmul_ST0_FT0_wrapper((void *)0ul, _var_977, *(generic64_t *)&_var_1843, *(generic16_t *)&_var_1842, *(generic64_t *)&_var_1841, *(generic16_t *)&_var_1840, *(generic64_t *)&_var_1839, *(generic16_t *)&_var_1838, *(generic64_t *)&_var_1837, *(generic16_t *)&_var_1836, *(generic64_t *)&_var_1835, *(generic16_t *)&_var_1834, *(generic64_t *)&_var_1833, *(generic16_t *)&_var_1832, *(generic64_t *)&_var_1831, *(generic16_t *)&_var_1830, *(generic64_t *)&_var_1829, *(generic16_t *)&_var_1828, 0u, 0u, _var_1827, 80u, 0u, 0u, _var_846, _var_847, (void *)&_var_829, (void *)&_var_830, (void *)&_var_831, (void *)&_var_832, (void *)&_var_833, (void *)&_var_834, (void *)&_var_835, (void *)&_var_836, (void *)&_var_837, (void *)&_var_838, (void *)&_var_839, (void *)&_var_840, (void *)&_var_841, (void *)&_var_842, (void *)&_var_843, (void *)&_var_844, (void *)&_var_845);
          *(generic64_t **)&_var_1403 = &_var_829;
          *(generic16_t **)&_var_1404 = &_var_830;
          *(generic64_t **)&_var_1405 = &_var_831;
          *(generic16_t **)&_var_1406 = &_var_832;
          *(generic64_t **)&_var_1407 = &_var_833;
          *(generic16_t **)&_var_1408 = &_var_834;
          *(generic64_t **)&_var_1409 = &_var_835;
          *(generic16_t **)&_var_1410 = &_var_836;
          *(generic64_t **)&_var_1411 = &_var_837;
          *(generic16_t **)&_var_1412 = &_var_838;
          *(generic64_t **)&_var_1413 = &_var_839;
          *(generic16_t **)&_var_1414 = &_var_840;
          *(generic64_t **)&_var_1415 = &_var_841;
          *(generic16_t **)&_var_1416 = &_var_842;
          *(generic64_t **)&_var_1417 = &_var_843;
          *(generic16_t **)&_var_1418 = &_var_844;
          *(generic8_t **)&_var_1419 = &_var_845;
          goto _label_1;
        }
      }
    } else {
      helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, *(generic32_t *)&_var_1710, *(generic64_t *)&_var_1711, *(generic16_t *)&_var_1712, *(generic64_t *)&_var_1713, *(generic16_t *)&_var_1714, *(generic64_t *)&_var_1715, *(generic16_t *)&_var_1716, *(generic64_t *)&_var_1717, *(generic16_t *)&_var_1718, *(generic64_t *)&_var_1719, *(generic16_t *)&_var_1720, *(generic64_t *)&_var_1721, *(generic16_t *)&_var_1722, *(generic64_t *)&_var_1723, *(generic16_t *)&_var_1724, *(generic64_t *)&_var_1725, *(generic16_t *)&_var_1726, (void *)&_var_1030, (void *)&_var_1031);
      helper_fucomi_ST0_FT0_wrapper((void *)0ul, (generic64_t)(uint64_t)(uint32_t)p, 24u, 97ul, 0ul, *(generic32_t *)&_var_1710, *(generic64_t *)&_var_1711, *(generic16_t *)&_var_1712, *(generic64_t *)&_var_1713, *(generic16_t *)&_var_1714, *(generic64_t *)&_var_1715, *(generic16_t *)&_var_1716, *(generic64_t *)&_var_1717, *(generic16_t *)&_var_1718, *(generic64_t *)&_var_1719, *(generic16_t *)&_var_1720, *(generic64_t *)&_var_1721, *(generic16_t *)&_var_1722, *(generic64_t *)&_var_1723, *(generic16_t *)&_var_1724, *(generic64_t *)&_var_1725, *(generic16_t *)&_var_1726, _var_1727, _var_1030, _var_1031, (void *)&_var_1028, (void *)&_var_1029);
      _var_1572 = _var_1029;
      _var_1556 = *(generic64_t *)&_var_1711;
      _var_1557 = *(generic16_t *)&_var_1712;
      _var_1558 = *(generic64_t *)&_var_1713;
      _var_1559 = *(generic16_t *)&_var_1714;
      _var_1560 = *(generic64_t *)&_var_1715;
      _var_1561 = *(generic16_t *)&_var_1716;
      _var_1562 = *(generic64_t *)&_var_1717;
      _var_1563 = *(generic16_t *)&_var_1718;
      _var_1564 = *(generic64_t *)&_var_1719;
      _var_1565 = *(generic16_t *)&_var_1720;
      _var_1566 = *(generic64_t *)&_var_1721;
      _var_1567 = *(generic16_t *)&_var_1722;
      _var_1568 = *(generic64_t *)&_var_1723;
      _var_1569 = *(generic16_t *)&_var_1724;
      _var_1570 = *(generic64_t *)&_var_1725;
      _var_1571 = *(generic16_t *)&_var_1726;
      if ((_var_1028 & 68ul) == 64ul) {
      } else {
        helper_flds_FT0_wrapper((void *)0ul, *(generic32_t *)4215996ul, _var_1029, 0u, 0u, (void *)&_var_974, (void *)&_var_975, (void *)&_var_976);
        helper_fmul_ST0_FT0_wrapper((void *)0ul, *(generic32_t *)&_var_1710, *(generic64_t *)&_var_1711, *(generic16_t *)&_var_1712, *(generic64_t *)&_var_1713, *(generic16_t *)&_var_1714, *(generic64_t *)&_var_1715, *(generic16_t *)&_var_1716, *(generic64_t *)&_var_1717, *(generic16_t *)&_var_1718, *(generic64_t *)&_var_1719, *(generic16_t *)&_var_1720, *(generic64_t *)&_var_1721, *(generic16_t *)&_var_1722, *(generic64_t *)&_var_1723, *(generic16_t *)&_var_1724, *(generic64_t *)&_var_1725, *(generic16_t *)&_var_1726, 0u, 0u, _var_974, 80u, 0u, 0u, _var_975, _var_976, (void *)&_var_957, (void *)&_var_958, (void *)&_var_959, (void *)&_var_960, (void *)&_var_961, (void *)&_var_962, (void *)&_var_963, (void *)&_var_964, (void *)&_var_965, (void *)&_var_966, (void *)&_var_967, (void *)&_var_968, (void *)&_var_969, (void *)&_var_970, (void *)&_var_971, (void *)&_var_972, (void *)&_var_973);
        _var_1556 = _var_957;
        _var_1557 = _var_958;
        _var_1558 = _var_959;
        _var_1559 = _var_960;
        _var_1560 = _var_961;
        _var_1561 = _var_962;
        _var_1562 = _var_963;
        _var_1563 = _var_964;
        _var_1564 = _var_965;
        _var_1565 = _var_966;
        _var_1566 = _var_967;
        _var_1567 = _var_968;
        _var_1568 = _var_969;
        _var_1569 = _var_970;
        _var_1570 = _var_971;
        _var_1571 = _var_972;
        _var_1572 = _var_973;
        ((generic32_t *)&_var_1346)[22ul] = ((generic32_t *)&_var_1346)[22ul] + 4294967268u;
      }
      _var_1539 = _var_1556;
      _var_1540 = _var_1557;
      _var_1541 = _var_1558;
      _var_1542 = _var_1559;
      _var_1543 = _var_1560;
      _var_1544 = _var_1561;
      _var_1545 = _var_1562;
      _var_1546 = _var_1563;
      _var_1547 = _var_1564;
      _var_1548 = _var_1565;
      _var_1549 = _var_1566;
      _var_1550 = _var_1567;
      _var_1551 = _var_1568;
      _var_1552 = _var_1569;
      _var_1553 = _var_1570;
      _var_1554 = _var_1571;
      _var_1555 = _var_1572;
      *(generic32_t *)&_var_1729 = ((generic32_t *)&_var_1346)[22ul];
      _var_1536 = (generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul);
      *(generic32_t *)&_var_1600 = helper_fnstcw_wrapper((void *)0ul, 895u);
      ((generic16_t *)&_var_1346)[39ul] = (generic16_t)*(generic32_t *)&_var_1600;
      ((generic16_t *)&_var_1346)[38ul] = (generic16_t)*(generic32_t *)&_var_1600 | 3072u;
      _var_1535 = 0ul;
      _var_1537 = *(generic32_t *)&_var_1710;
      _var_1538 = 895u;
    _label_2:
      *(generic64_t *)&_var_1730 = _var_1535;
      *(generic64_t *)&_var_1731 = _var_1536;
      helper_fpush_wrapper((void *)0ul, _var_1537, (void *)&_var_804, (void *)&_var_805, (void *)&_var_806, (void *)&_var_807, (void *)&_var_808, (void *)&_var_809, (void *)&_var_810, (void *)&_var_811, (void *)&_var_812);
      *(generic32_t *)&_var_1732 = _var_804;
      helper_fmov_ST0_STN_wrapper((void *)0ul, 1u, *(generic32_t *)&_var_1732, _var_1539, _var_1540, _var_1541, _var_1542, _var_1543, _var_1544, _var_1545, _var_1546, _var_1547, _var_1548, _var_1549, _var_1550, _var_1551, _var_1552, _var_1553, _var_1554, (void *)&_var_788, (void *)&_var_789, (void *)&_var_790, (void *)&_var_791, (void *)&_var_792, (void *)&_var_793, (void *)&_var_794, (void *)&_var_795, (void *)&_var_796, (void *)&_var_797, (void *)&_var_798, (void *)&_var_799, (void *)&_var_800, (void *)&_var_801, (void *)&_var_802, (void *)&_var_803);
      helper_fldcw_wrapper((void *)0ul, (generic32_t)(uint32_t)(uint16_t)((generic16_t *)&_var_1346)[38ul], _var_1538, (void *)&_var_785, (void *)&_var_786, (void *)&_var_787);
      *(generic16_t *)&_var_1733 = _var_785;
      *(generic64_t *)&_var_1601 = helper_fistll_ST0_wrapper((void *)0ul, *(generic32_t *)&_var_1732, _var_788, _var_789, _var_790, _var_791, _var_792, _var_793, _var_794, _var_795, _var_796, _var_797, _var_798, _var_799, _var_800, _var_801, _var_802, _var_803, _var_786, _var_1555, (void *)&_var_784);
      _var_1734 = _var_784;
      ((generic64_t *)&_var_1346)[7ul] = *(generic64_t *)&_var_1601;
      helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1732, (void *)&_var_775, (void *)&_var_776, (void *)&_var_777, (void *)&_var_778, (void *)&_var_779, (void *)&_var_780, (void *)&_var_781, (void *)&_var_782, (void *)&_var_783);
      *(generic32_t *)&_var_1735 = _var_775;
      helper_fldcw_wrapper((void *)0ul, (generic32_t)(uint32_t)(uint16_t)((generic16_t *)&_var_1346)[39ul], *(generic16_t *)&_var_1733, (void *)&_var_772, (void *)&_var_773, (void *)&_var_774);
      _var_1538 = _var_772;
      _var_1736 = _var_773;
      _var_1737 = _var_774;
      _var_1536 = *(generic64_t *)&_var_1731 + 4ul;
      *(generic64_t *)&_var_1738 = ((generic64_t *)&_var_1346)[7ul];
      *(generic32_t *)*(generic64_t *)&_var_1731 = (generic32_t)*(generic64_t *)&_var_1738;
      ((generic64_t *)&_var_1346)[7ul] = *(generic64_t *)&_var_1738 & 4294967295ul;
      helper_fildll_ST0_wrapper((void *)0ul, *(generic64_t *)&_var_1738 & 4294967295ul, *(generic32_t *)&_var_1735, (void *)&_var_747, (void *)&_var_748, (void *)&_var_749, (void *)&_var_750, (void *)&_var_751, (void *)&_var_752, (void *)&_var_753, (void *)&_var_754, (void *)&_var_755, (void *)&_var_756, (void *)&_var_757, (void *)&_var_758, (void *)&_var_759, (void *)&_var_760, (void *)&_var_761, (void *)&_var_762, (void *)&_var_763, (void *)&_var_764, (void *)&_var_765, (void *)&_var_766, (void *)&_var_767, (void *)&_var_768, (void *)&_var_769, (void *)&_var_770, (void *)&_var_771);
      helper_fsub_STN_ST0_wrapper((void *)0ul, 1u, _var_747, _var_756, _var_757, _var_758, _var_759, _var_760, _var_761, _var_762, _var_763, _var_764, _var_765, _var_766, _var_767, _var_768, _var_769, _var_770, _var_771, 0u, _var_1736, _var_1734, _var_1737, 0u, 0u, (void *)&_var_730, (void *)&_var_731, (void *)&_var_732, (void *)&_var_733, (void *)&_var_734, (void *)&_var_735, (void *)&_var_736, (void *)&_var_737, (void *)&_var_738, (void *)&_var_739, (void *)&_var_740, (void *)&_var_741, (void *)&_var_742, (void *)&_var_743, (void *)&_var_744, (void *)&_var_745, (void *)&_var_746);
      helper_fpop_wrapper((void *)0ul, _var_747, (void *)&_var_721, (void *)&_var_722, (void *)&_var_723, (void *)&_var_724, (void *)&_var_725, (void *)&_var_726, (void *)&_var_727, (void *)&_var_728, (void *)&_var_729);
      _var_1537 = _var_721;
      helper_flds_FT0_wrapper((void *)0ul, *(generic32_t *)4216000ul, _var_746, 0u, 0u, (void *)&_var_718, (void *)&_var_719, (void *)&_var_720);
      helper_fmul_ST0_FT0_wrapper((void *)0ul, _var_1537, _var_730, _var_731, _var_732, _var_733, _var_734, _var_735, _var_736, _var_737, _var_738, _var_739, _var_740, _var_741, _var_742, _var_743, _var_744, _var_745, 0u, _var_1736, _var_718, _var_1737, 0u, 0u, _var_719, _var_720, (void *)&_var_701, (void *)&_var_702, (void *)&_var_703, (void *)&_var_704, (void *)&_var_705, (void *)&_var_706, (void *)&_var_707, (void *)&_var_708, (void *)&_var_709, (void *)&_var_710, (void *)&_var_711, (void *)&_var_712, (void *)&_var_713, (void *)&_var_714, (void *)&_var_715, (void *)&_var_716, (void *)&_var_717);
      _var_1539 = _var_701;
      _var_1540 = _var_702;
      _var_1541 = _var_703;
      _var_1542 = _var_704;
      _var_1543 = _var_705;
      _var_1544 = _var_706;
      _var_1545 = _var_707;
      _var_1546 = _var_708;
      _var_1547 = _var_709;
      _var_1548 = _var_710;
      _var_1549 = _var_711;
      _var_1550 = _var_712;
      _var_1551 = _var_713;
      _var_1552 = _var_714;
      _var_1553 = _var_715;
      _var_1554 = _var_716;
      helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, _var_1537, _var_1539, _var_1540, _var_1541, _var_1542, _var_1543, _var_1544, _var_1545, _var_1546, _var_1547, _var_1548, _var_1549, _var_1550, _var_1551, _var_1552, _var_1553, _var_1554, (void *)&_var_699, (void *)&_var_700);
      helper_fucomi_ST0_FT0_wrapper((void *)0ul, ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + (generic64_t)&_var_1346 + 7532ul + (*(generic64_t *)&_var_1730 << 2ul), 9u, 4ul, 0ul, _var_1537, _var_1539, _var_1540, _var_1541, _var_1542, _var_1543, _var_1544, _var_1545, _var_1546, _var_1547, _var_1548, _var_1549, _var_1550, _var_1551, _var_1552, _var_1553, _var_1554, _var_717, _var_699, _var_700, (void *)&_var_697, (void *)&_var_698);
      _var_1739 = _var_698;
      _var_1555 = _var_1739;
      _var_1535 = *(generic64_t *)&_var_1730 + 1ul;
      if ((_var_697 & 68ul) == 64ul) {
        helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_721, _var_701, _var_702, _var_703, _var_704, _var_705, _var_706, _var_707, _var_708, _var_709, _var_710, _var_711, _var_712, _var_713, _var_714, _var_715, _var_716, (void *)&_var_458, (void *)&_var_459, (void *)&_var_460, (void *)&_var_461, (void *)&_var_462, (void *)&_var_463, (void *)&_var_464, (void *)&_var_465, (void *)&_var_466, (void *)&_var_467, (void *)&_var_468, (void *)&_var_469, (void *)&_var_470, (void *)&_var_471, (void *)&_var_472, (void *)&_var_473);
        helper_fpop_wrapper((void *)0ul, _var_721, (void *)&_var_449, (void *)&_var_450, (void *)&_var_451, (void *)&_var_452, (void *)&_var_453, (void *)&_var_454, (void *)&_var_455, (void *)&_var_456, (void *)&_var_457);
        helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_449, _var_458, _var_459, _var_460, _var_461, _var_462, _var_463, _var_464, _var_465, _var_466, _var_467, _var_468, _var_469, _var_470, _var_471, _var_472, _var_473, (void *)&_var_433, (void *)&_var_434, (void *)&_var_435, (void *)&_var_436, (void *)&_var_437, (void *)&_var_438, (void *)&_var_439, (void *)&_var_440, (void *)&_var_441, (void *)&_var_442, (void *)&_var_443, (void *)&_var_444, (void *)&_var_445, (void *)&_var_446, (void *)&_var_447, (void *)&_var_448);
        helper_fpop_wrapper((void *)0ul, _var_449, (void *)&_var_424, (void *)&_var_425, (void *)&_var_426, (void *)&_var_427, (void *)&_var_428, (void *)&_var_429, (void *)&_var_430, (void *)&_var_431, (void *)&_var_432);
        *(generic32_t *)&_var_1740 = _var_424;
        *(generic64_t *)&_var_1602 = lshift((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1729, 4294967272u);
        _var_1521 = ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + (generic64_t)&_var_1346 + 7532ul + (*(generic64_t *)&_var_1730 << 2ul);
        _var_1522 = (generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul);
        _var_1523 = *(generic32_t *)&_var_1729;
        if (((*(generic32_t *)&_var_1729 == 0u ? 64u : 0u) | (generic8_t)*(generic64_t *)&_var_1602 & 128u) == 0u) {
          _var_1531 = (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)&_var_1729;
          _var_1532 = *(generic32_t *)&_var_1729;
          _var_1533 = (generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul);
          _var_1534 = ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + (generic64_t)&_var_1346 + 7532ul + (*(generic64_t *)&_var_1730 << 2ul);
        _label_3:
          *(generic32_t *)&_var_1742 = _var_1532;
          *(generic64_t *)&_var_1743 = _var_1533;
          *(generic64_t *)&_var_1744 = _var_1534;
          *(generic64_t *)&_var_1741 = (generic8_t)((int32_t)*(generic32_t *)&_var_1742 > 29) ? 29ul : _var_1531;
          _var_1527 = 0ul;
          if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1743 > (uint64_t)(*(generic64_t *)&_var_1744 + 18446744073709551612ul))) {
            _var_1526 = *(generic64_t *)&_var_1743;
            if ((generic32_t)_var_1527 == 0u) {
            } else {
              _var_1526 = *(generic64_t *)&_var_1743 + 18446744073709551612ul;
              *(generic32_t *)_var_1526 = (generic32_t)_var_1527;
            }
            _var_1524 = 0ul;
            _var_1525 = *(generic64_t *)&_var_1744;
          _label_4:
            *(generic64_t *)&_var_1749 = _var_1525;
            if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1749 > (uint64_t)_var_1526)) {
              _var_1525 = *(generic64_t *)&_var_1749 + 18446744073709551612ul;
              *(int8_t *)&_var_1748 = *(generic32_t *)(*(generic64_t *)&_var_1744 + 18446744073709551612ul - (_var_1524 << 2ul)) == 0u;
              _var_1524 = _var_1524 + 1ul;
              if (_var_1748) {
                goto _label_4;
              } else {
                _var_1532 = *(generic32_t *)&_var_1742 - (generic32_t)*(generic64_t *)&_var_1741;
                _var_1531 = (generic64_t)(uint64_t)(uint32_t)_var_1532;
                *(generic64_t *)&_var_1603 = lshift(_var_1531, 4294967272u);
                if (((_var_1532 == 0u ? 64u : 0u) | (generic8_t)*(generic64_t *)&_var_1603 & 128u) == 0u) {
                  goto _label_3;
                } else {
                  _var_1521 = *(generic64_t *)&_var_1749;
                  _var_1522 = _var_1526;
                  _var_1523 = *(generic32_t *)&_var_1742 - (generic32_t)*(generic64_t *)&_var_1741;
                  *(generic64_t *)&_var_1750 = _var_1521;
                  *(generic64_t *)&_var_1751 = _var_1522;
                  if (((*(generic32_t *)&_var_1729 == 0u ? 64u : 0u) | (generic8_t)*(generic64_t *)&_var_1602 & 128u) == 0u) {
                    ((generic32_t *)&_var_1346)[22ul] = _var_1523;
                  } else {
                  }
                  *(generic32_t *)&_var_1752 = ((generic32_t *)&_var_1346)[22ul];
                  _var_1508 = *(generic32_t *)&_var_1752;
                  *(generic32_t *)&_var_1753 = _var_1508;
                  ((generic64_t *)&_var_1346)[8ul] = (generic64_t)((int64_t)(((generic64_t)((int64_t)(((generic8_t)((int32_t)((generic32_t)((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) + 29u) > 4294967295) ? 0ul : 18446744069414584320ul) | (generic64_t)(uint64_t)(uint32_t)((generic32_t)((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) + 29u)) / 9l) << 32ul) + 4294967296ul) >> 30l);
                  _var_1506 = *(generic64_t *)&_var_1750;
                  _var_1507 = *(generic64_t *)&_var_1751;
                  if ((generic8_t)((int32_t)*(generic32_t *)&_var_1753 > 4294967295)) {
                    *(generic64_t *)&_var_1762 = _var_1506;
                    *(generic64_t *)&_var_1763 = _var_1507;
                    if ((generic8_t)((int32_t)*(generic32_t *)&_var_1753 > 4294967295)) {
                    } else {
                      ((generic32_t *)&_var_1346)[22ul] = _var_1508;
                    }
                    _var_1503 = 0ul;
                    if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1763 < (uint64_t)*(generic64_t *)&_var_1762)) {
                      *(generic32_t *)&_var_1764 = *(generic32_t *)*(generic64_t *)&_var_1763;
                      _var_1503 = (generic64_t)((uint64_t)((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) - *(generic64_t *)&_var_1763) >> 2ul) * 9ul & 4294967295ul;
                      if ((generic8_t)((uint32_t)*(generic32_t *)&_var_1764 < 10u)) {
                        *(generic64_t *)&_var_1765 = _var_1503;
                        _var_1451 = *(generic64_t *)&_var_1765;
                        _var_1449 = *(generic64_t *)&_var_1762;
                        _var_1450 = *(generic64_t *)&_var_1763;
                        if ((generic8_t)((int64_t)((((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) - (*(generic32_t *)&_var_1728 == 102u ? 0ul : _var_1451) & 4294967295ul) + (generic64_t)(int64_t)(int8_t)((generic8_t)(*(generic32_t *)&_var_1728 == 103u) & (generic8_t)(((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) != 0ul)) << 32ul) >> 32l < (int64_t)((generic64_t)((int64_t)(*(generic64_t *)&_var_1762 - ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul))) >> 2l) * 9ul + 18446744073709551607ul))) {
                          *(generic64_t *)&_var_1766 = (generic64_t)((int64_t)(((generic8_t)((int32_t)((generic32_t)((((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) - (*(generic32_t *)&_var_1728 == 102u ? 0ul : _var_1451) & 4294967295ul) + (generic64_t)(int64_t)(int8_t)((generic8_t)(*(generic32_t *)&_var_1728 == 103u) & (generic8_t)(((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) != 0ul))) + 147456u) > 4294967295) ? 0ul : 18446744069414584320ul) | (generic64_t)(uint64_t)(uint32_t)((generic32_t)((((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) - (*(generic32_t *)&_var_1728 == 102u ? 0ul : _var_1451) & 4294967295ul) + (generic64_t)(int64_t)(int8_t)((generic8_t)(*(generic32_t *)&_var_1728 == 103u) & (generic8_t)(((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) != 0ul))) + 147456u)) / 9l) << 2ul;
                          _var_1500 = 10ul;
                          if (((generic64_t)((int64_t)(((generic8_t)((int32_t)((generic32_t)((((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) - (*(generic32_t *)&_var_1728 == 102u ? 0ul : _var_1451) & 4294967295ul) + (generic64_t)(int64_t)(int8_t)((generic8_t)(*(generic32_t *)&_var_1728 == 103u) & (generic8_t)(((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) != 0ul))) + 147456u) > 4294967295) ? 0ul : 18446744069414584320ul) | (generic64_t)(uint64_t)(uint32_t)((generic32_t)((((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) - (*(generic32_t *)&_var_1728 == 102u ? 0ul : _var_1451) & 4294967295ul) + (generic64_t)(int64_t)(int8_t)((generic8_t)(*(generic32_t *)&_var_1728 == 103u) & (generic8_t)(((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) != 0ul))) + 147456u)) % 9l) & 4294967295ul) == 8ul) {
                            *(generic64_t *)&_var_1767 = (generic64_t)(uint64_t)(uint32_t)((generic32_t *)(*(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul))))[4611686018427371521ul];
                            _var_1452 = *(generic64_t *)&_var_1763;
                            _var_1453 = *(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul)) + 18446744073709486084ul;
                            _var_1454 = *(generic64_t *)&_var_1765;
                            if ((generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500) == 0ul ? (generic8_t)(*(generic64_t *)&_var_1762 == *(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul)) + 18446744073709486088ul) : 0u) {
                              _var_1450 = _var_1452;
                              _var_1451 = _var_1454;
                              _var_1449 = llvm.umin.i64(*(generic64_t *)&_var_1762, _var_1453 + 4ul);
                              _var_1448 = _var_1449;
                              *(generic64_t *)&_var_1776 = _var_1450;
                              *(generic64_t *)&_var_1777 = _var_1451;
                              *(generic64_t *)&_var_1775 = _var_1448 + 18446744073709551612ul;
                              _var_1447 = 0ul;
                            _label_5:
                              *(generic64_t *)&_var_1779 = _var_1448;
                              if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1779 > (uint64_t)*(generic64_t *)&_var_1776)) {
                                _var_1448 = *(generic64_t *)&_var_1779 + 18446744073709551612ul;
                                *(int8_t *)&_var_1778 = *(generic32_t *)(*(generic64_t *)&_var_1775 - (_var_1447 << 2ul)) == 0u;
                                _var_1447 = _var_1447 + 1ul;
                                if (_var_1778) {
                                  goto _label_5;
                                } else {
                                  _var_1441 = (generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul;
                                  if (*(generic32_t *)&_var_1728 == 103u) {
                                    if ((generic8_t)((int64_t)((((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) == 0ul ? 1ul : (generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) << 32ul) <= (int64_t)(*(generic64_t *)&_var_1777 << 32ul)) | (generic8_t)((int32_t)(generic32_t)*(generic64_t *)&_var_1777 < 4294967292)) {
                                      _var_1445 = ((generic32_t *)&_var_1346)[6ul] + 4294967294u;
                                      ((generic32_t *)&_var_1346)[6ul] = _var_1445;
                                      _var_1446 = (((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) == 0ul ? 1ul : (generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) + 4294967295ul;
                                    } else {
                                      _var_1445 = ((generic32_t *)&_var_1346)[6ul] + 4294967295u;
                                      ((generic32_t *)&_var_1346)[6ul] = _var_1445;
                                      _var_1446 = (((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) == 0ul ? 1ul : (generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) - (*(generic64_t *)&_var_1777 + 1ul & 4294967295ul);
                                    }
                                    *(int8_t *)&_var_1780 = (((generic8_t *)&_var_1346)[20ul] & 8u) == 0u;
                                    _var_1441 = _var_1446 & 4294967295ul;
                                    if (_var_1780) {
                                      _var_1442 = 9ul;
                                      if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1779 > (uint64_t)*(generic64_t *)&_var_1776)) {
                                        *(generic32_t *)&_var_1781 = ((generic32_t *)*(generic64_t *)&_var_1779)[4611686018427387903ul];
                                        _var_1442 = 9ul;
                                        if (*(generic32_t *)&_var_1781 == 0u) {
                                          _var_1441 = llvm.smin.i64(llvm.smax.i64((generic64_t)((int64_t)(*(generic64_t *)&_var_1779 - ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul))) >> 2l) * 9ul + 18446744073709551607ul + ((_var_1445 & 4294967263u) == 70u ? 0ul : (generic64_t)((int64_t)(*(generic64_t *)&_var_1777 << 32ul) >> 32l)) - _var_1442, 0ul), (generic64_t)((int64_t)(_var_1446 << 32ul) >> 32l));
                                          *(generic64_t *)&_var_1783 = _var_1441;
                                          ((generic32_t *)&_var_1346)[14ul] = ((generic32_t *)&_var_1346)[5ul] & 8u;
                                          ((generic32_t *)&_var_1346)[18ul] = ((generic32_t *)&_var_1346)[14ul] | (generic32_t)*(generic64_t *)&_var_1783;
                                          ((generic32_t *)&_var_1346)[16ul] = ((generic32_t *)&_var_1346)[6ul] | 32u;
                                          if (((generic32_t *)&_var_1346)[16ul] == 102u) {
                                            *(generic64_t *)&_var_1609 = lshift(*(generic64_t *)&_var_1777 & 4294967295ul, 4294967272u);
                                            _var_1437 = (((*(generic64_t *)&_var_1777 & 4294967295ul) == 0ul ? 64u : 0u) | (generic8_t)*(generic64_t *)&_var_1609 & 128u) == 0u ? *(generic64_t *)&_var_1777 : 0ul;
                                            *(generic32_t *)&_var_1788 = ((generic32_t *)&_var_1346)[4ul];
                                            *(generic32_t *)&_var_1789 = ((generic32_t *)&_var_1346)[5ul];
                                            ((generic32_t *)&_var_1346)[6ul] = ((generic32_t *)&_var_1346)[12ul] + (generic32_t)(*(generic64_t *)&_var_1783 + (generic64_t)(uint64_t)(uint8_t)((((generic32_t *)&_var_1346)[14ul] | (generic32_t)*(generic64_t *)&_var_1783) != 0u) + 1ul + _var_1437);
                                            pad(f, 32, (int32_t)*(generic32_t *)&_var_1788, (int32_t)((generic32_t *)&_var_1346)[6ul], (int32_t)*(generic32_t *)&_var_1789);
                                            out(f, (const int8_t *)*(generic64_t *)&_var_1670, (size_t)(int64_t)(int32_t)((generic32_t *)&_var_1346)[12ul]);
                                            pad(f, 48, (int32_t)((generic32_t *)&_var_1346)[4ul], (int32_t)((generic32_t *)&_var_1346)[6ul], (int32_t)(((generic32_t *)&_var_1346)[5ul] ^ 65536u));
                                            if (((generic32_t *)&_var_1346)[16ul] == 102u) {
                                              generic64_t _var_1913 = llvm.umin.i64((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul), *(generic64_t *)&_var_1776);
                                              _var_1429 = _var_1913;
                                              *(generic64_t *)&_var_1798 = _var_1429 + 4ul;
                                              _var_1428 = 0ul;
                                            _label_6:
                                              *(generic64_t *)&_var_1799 = _var_1428;
                                              *(generic64_t *)&_var_1800 = _var_1429;
                                              *(int8_t **)&_var_1611 = fmt_u((unreserved_uintmax_t)(uint64_t)(uint32_t)*(generic32_t *)*(generic64_t *)&_var_1800, (int8_t *)&_var_1346 + 116ul);
                                              if (*(generic64_t *)&_var_1800 == _var_1913) {
                                                _var_1426 = *(generic64_t *)&_var_1611;
                                                if (*(generic64_t *)&_var_1611 == (generic64_t)&_var_1346 + 116ul) {
                                                  ((generic8_t *)&_var_1346)[115ul] = 48u;
                                                  _var_1426 = (generic64_t)&_var_1346 + 115ul;
                                                } else {
                                                }
                                                _var_1429 = *(generic64_t *)&_var_1800 + 4ul;
                                                out(f, (const int8_t *)_var_1426, (size_t)((generic64_t)&_var_1346 + 116ul - _var_1426));
                                                _var_1428 = *(generic64_t *)&_var_1799 + 1ul;
                                                if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_1798 + (*(generic64_t *)&_var_1799 << 2ul)) > (uint64_t)((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul)))) {
                                                  *(int8_t *)&_var_1802 = ((generic32_t *)&_var_1346)[18ul] == 0u;
                                                  _var_1420 = 0ul;
                                                  if (_var_1802) {
                                                    pad(f, 48, (int32_t)((generic32_t)_var_1420 + 9u), 9, 0);
                                                    pad(f, 32, (int32_t)((generic32_t *)&_var_1346)[4ul], (int32_t)((generic32_t *)&_var_1346)[6ul], (int32_t)(((generic32_t *)&_var_1346)[5ul] ^ 8192u));
                                                    _var_1347 = llvm.smax.i32(((generic32_t *)&_var_1346)[4ul], ((generic32_t *)&_var_1346)[6ul]);
                                                  } else {
                                                    out(f, (const int8_t *)4215955ul, 1ul);
                                                    _var_1420 = *(generic64_t *)&_var_1783;
                                                    if ((generic8_t)((uint64_t)(((generic8_t)((uint64_t)((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) | 1ul) < (uint64_t)(_var_1913 + 18446744073709551613ul)) ? 0ul : (generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + 4ul - _var_1913 & 18446744073709551612ul) + _var_1913) < (uint64_t)*(generic64_t *)&_var_1779)) {
                                                      _var_1423 = 0ul;
                                                      _var_1424 = *(generic64_t *)&_var_1783;
                                                      _var_1425 = ((generic8_t)((uint64_t)((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) | 1ul) < (uint64_t)(_var_1913 + 18446744073709551613ul)) ? 0ul : (generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + 4ul - _var_1913 & 18446744073709551612ul) + _var_1913;
                                                    _label_7:
                                                      *(generic64_t *)&_var_1803 = _var_1423;
                                                      *(generic64_t *)&_var_1804 = _var_1424;
                                                      _var_1421 = *(generic64_t *)&_var_1804;
                                                      *(generic64_t *)&_var_1806 = _var_1425;
                                                      *(generic32_t *)&_var_1805 = (generic32_t)_var_1421;
                                                      *(generic64_t *)&_var_1612 = lshift(_var_1421 & 4294967295ul, 4294967272u);
                                                      if (((*(generic32_t *)&_var_1805 == 0u ? 64u : 0u) | (generic8_t)*(generic64_t *)&_var_1612 & 128u) == 0u) {
                                                        *(int8_t **)&_var_1613 = fmt_u((unreserved_uintmax_t)(uint64_t)(uint32_t)*(generic32_t *)*(generic64_t *)&_var_1806, (int8_t *)&_var_1346 + 116ul);
                                                        if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1613 > (uint64_t)((generic64_t)&_var_1346 + 107ul))) {
                                                          _var_1422 = 0ul;
                                                        _label_8:
                                                          *(generic64_t *)&_var_1807 = _var_1422;
                                                          *(generic8_t *)(*(generic64_t *)&_var_1613 + 18446744073709551615ul - *(generic64_t *)&_var_1807) = 48u;
                                                          _var_1422 = *(generic64_t *)&_var_1807 + 1ul;
                                                          if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_1613 + 18446744073709551615ul - *(generic64_t *)&_var_1807) > (uint64_t)((generic64_t)&_var_1346 + 107ul))) {
                                                            goto _label_8;
                                                          } else {
                                                            _var_1424 = *(generic64_t *)&_var_1804 + 4294967287ul & 4294967295ul;
                                                            _var_1421 = _var_1424;
                                                            _var_1425 = *(generic64_t *)&_var_1806 + 4ul;
                                                            out(f, (const int8_t *)llvm.umin.i64(*(generic64_t *)&_var_1613, (generic64_t)&_var_1346 + 107ul), (size_t)((generic8_t)((int32_t)*(generic32_t *)&_var_1805 > 9) ? 9ul : (generic64_t)((int64_t)(*(generic64_t *)&_var_1804 << 32ul) >> 32l)));
                                                            _var_1423 = *(generic64_t *)&_var_1803 + 1ul;
                                                            if ((generic8_t)((uint64_t)(((generic8_t)((uint64_t)((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) | 1ul) < (uint64_t)(_var_1913 + 18446744073709551613ul)) ? 0ul : (generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + 4ul - _var_1913 & 18446744073709551612ul) + _var_1913 + 4ul + (*(generic64_t *)&_var_1803 << 2ul)) < (uint64_t)*(generic64_t *)&_var_1779)) {
                                                              goto _label_7;
                                                            } else {
                                                              _var_1420 = _var_1421;
                                                            }
                                                          }
                                                        } else {
                                                        }
                                                      } else {
                                                      }
                                                    } else {
                                                    }
                                                  }
                                                } else {
                                                  goto _label_6;
                                                }
                                              } else if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1611 > (uint64_t)((generic64_t)&_var_1346 + 107ul))) {
                                                _var_1427 = 0ul;
                                              _label_9:
                                                *(generic64_t *)&_var_1801 = _var_1427;
                                                *(generic8_t *)(*(generic64_t *)&_var_1611 + 18446744073709551615ul - *(generic64_t *)&_var_1801) = 48u;
                                                _var_1427 = *(generic64_t *)&_var_1801 + 1ul;
                                                if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_1611 + 18446744073709551615ul - *(generic64_t *)&_var_1801) > (uint64_t)((generic64_t)&_var_1346 + 107ul))) {
                                                  goto _label_9;
                                                } else {
                                                  _var_1426 = llvm.umin.i64(*(generic64_t *)&_var_1611, (generic64_t)&_var_1346 + 107ul);
                                                }
                                              } else {
                                              }
                                            } else {
                                              _var_1430 = *(generic64_t *)&_var_1783;
                                              if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1776 < (uint64_t)((generic8_t)((uint64_t)*(generic64_t *)&_var_1779 > (uint64_t)*(generic64_t *)&_var_1776) ? *(generic64_t *)&_var_1779 : *(generic64_t *)&_var_1776 + 4ul)) ? (generic8_t)((int32_t)(generic32_t)*(generic64_t *)&_var_1783 > 4294967295) : 0u) {
                                                _var_1433 = 0ul;
                                                _var_1434 = (generic32_t)*(generic64_t *)&_var_1783;
                                                _var_1435 = *(generic64_t *)&_var_1783;
                                                _var_1436 = *(generic64_t *)&_var_1776;
                                              _label_10:
                                                *(generic64_t *)&_var_1790 = _var_1433;
                                                *(generic32_t *)&_var_1791 = _var_1434;
                                                *(generic64_t *)&_var_1792 = _var_1435;
                                                *(generic64_t *)&_var_1793 = _var_1436;
                                                *(int8_t **)&_var_1610 = fmt_u((unreserved_uintmax_t)(uint64_t)(uint32_t)*(generic32_t *)*(generic64_t *)&_var_1793, (int8_t *)&_var_1346 + 116ul);
                                                _var_1432 = *(generic64_t *)&_var_1610;
                                                if (_var_1432 == (generic64_t)&_var_1346 + 116ul) {
                                                  ((generic8_t *)&_var_1346)[115ul] = 48u;
                                                  _var_1432 = (generic64_t)&_var_1346 + 115ul;
                                                } else {
                                                }
                                                *(generic64_t *)&_var_1794 = _var_1432;
                                                if (*(generic64_t *)&_var_1793 == *(generic64_t *)&_var_1776) {
                                                  ((generic64_t *)&_var_1346)[6ul] = *(generic64_t *)&_var_1794 + 1ul;
                                                  out(f, (const int8_t *)*(generic64_t *)&_var_1794, 1ul);
                                                  if ((((generic32_t *)&_var_1346)[14ul] | *(generic32_t *)&_var_1791) == 0u) {
                                                  } else {
                                                    out(f, (const int8_t *)4215955ul, 1ul);
                                                  }
                                                  *(generic64_t *)&_var_1796 = ((generic64_t *)&_var_1346)[6ul];
                                                  ((generic64_t *)&_var_1346)[8ul] = (generic64_t)&_var_1346 + 116ul - *(generic64_t *)&_var_1796;
                                                  _var_1436 = *(generic64_t *)&_var_1793 + 4ul;
                                                  out(f, (const int8_t *)*(generic64_t *)&_var_1796, (size_t)llvm.smin.i64((generic64_t)((int64_t)(*(generic64_t *)&_var_1792 << 32ul) >> 32l), (generic64_t)&_var_1346 + 116ul - *(generic64_t *)&_var_1796));
                                                  *(generic64_t *)&_var_1797 = *(generic64_t *)&_var_1792 - ((generic64_t *)&_var_1346)[8ul];
                                                  _var_1435 = *(generic64_t *)&_var_1797 & 4294967295ul;
                                                  _var_1434 = (generic32_t)*(generic64_t *)&_var_1797;
                                                  _var_1433 = *(generic64_t *)&_var_1790 + 1ul;
                                                  if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_1776 + 4ul + (*(generic64_t *)&_var_1790 << 2ul)) < (uint64_t)((generic8_t)((uint64_t)*(generic64_t *)&_var_1779 > (uint64_t)*(generic64_t *)&_var_1776) ? *(generic64_t *)&_var_1779 : *(generic64_t *)&_var_1776 + 4ul)) ? (generic8_t)((int32_t)_var_1434 > 4294967295) : 0u) {
                                                    goto _label_10;
                                                  } else {
                                                    _var_1430 = *(generic64_t *)&_var_1797 & 4294967295ul;
                                                    pad(f, 48, (int32_t)((generic32_t)_var_1430 + 18u), 18, 0);
                                                    out(f, (const int8_t *)((generic64_t *)&_var_1346)[4ul], (size_t)((generic64_t)&_var_1346 + 107ul - ((generic64_t *)&_var_1346)[4ul]));
                                                  }
                                                } else if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1794 > (uint64_t)((generic64_t)&_var_1346 + 107ul))) {
                                                  _var_1431 = 0ul;
                                                _label_11:
                                                  *(generic64_t *)&_var_1795 = _var_1431;
                                                  *(generic8_t *)(*(generic64_t *)&_var_1794 + 18446744073709551615ul - *(generic64_t *)&_var_1795) = 48u;
                                                  _var_1431 = *(generic64_t *)&_var_1795 + 1ul;
                                                  if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_1794 + 18446744073709551615ul - *(generic64_t *)&_var_1795) > (uint64_t)((generic64_t)&_var_1346 + 107ul))) {
                                                    goto _label_11;
                                                  } else {
                                                    ((generic64_t *)&_var_1346)[6ul] = llvm.umin.i64(*(generic64_t *)&_var_1794, (generic64_t)&_var_1346 + 107ul);
                                                  }
                                                } else {
                                                }
                                              } else {
                                              }
                                            }
                                          } else {
                                            *(int8_t **)&_var_1606 = fmt_u((unreserved_uintmax_t)((int64_t)((((*(generic64_t *)&_var_1777 & 2147483648ul) == 0ul ? 0ul : 4294967295ul) ^ *(generic64_t *)&_var_1777) - ((*(generic64_t *)&_var_1777 & 2147483648ul) == 0ul ? 0ul : 4294967295ul) << 32ul) >> 32l), (int8_t *)&_var_1346 + 107ul);
                                            _var_1438 = *(generic64_t *)&_var_1606;
                                            *(generic64_t *)&_var_1607 = lshift((generic64_t)&_var_1346 + 107ul + (_var_1438 ^ 18446744073709551615ul), 4294967240u);
                                            if ((((generic64_t)&_var_1346 + 107ul + (_var_1438 ^ 18446744073709551615ul) == 0ul ? 64u : 0u) | (generic8_t)*(generic64_t *)&_var_1607 & 128u) == ((generic64_t)&_var_1346 + 107ul + (_var_1438 ^ 18446744073709551615ul) == 9223372036854775807ul ? 128u : 0u)) {
                                              *(generic64_t *)&_var_1786 = _var_1438;
                                              ((generic64_t *)&_var_1346)[4ul] = *(generic64_t *)&_var_1786 + 18446744073709551614ul;
                                              _var_1787 = ((generic8_t *)&_var_1346)[24ul];
                                              ((generic8_t *)*(generic64_t *)&_var_1786)[18446744073709551615ul] = ((generic8_t)(generic64_t)((uint64_t)*(generic64_t *)&_var_1777 >> 30ul) & 2u) + 43u;
                                              ((generic8_t *)*(generic64_t *)&_var_1786)[18446744073709551614ul] = _var_1787;
                                              _var_1437 = (generic64_t)&_var_1346 + 4294967403ul - ((generic64_t *)&_var_1346)[4ul];
                                            } else {
                                              _var_1439 = 0ul;
                                              _var_1440 = *(generic64_t *)&_var_1606;
                                            _label_12:
                                              *(generic64_t *)&_var_1784 = _var_1439;
                                              *(generic64_t *)&_var_1785 = _var_1440;
                                              _var_1440 = *(generic64_t *)&_var_1785 + 18446744073709551615ul;
                                              *(generic8_t *)(*(generic64_t *)&_var_1606 + 18446744073709551615ul - *(generic64_t *)&_var_1784) = 48u;
                                              *(generic64_t *)&_var_1608 = lshift((generic64_t)&_var_1346 + 107ul - *(generic64_t *)&_var_1606 + *(generic64_t *)&_var_1784, 4294967240u);
                                              _var_1439 = *(generic64_t *)&_var_1784 + 1ul;
                                              if ((((generic64_t)&_var_1346 + 107ul == *(generic64_t *)&_var_1785 ? 64u : 0u) | (generic8_t)*(generic64_t *)&_var_1608 & 128u) == ((generic64_t)&_var_1346 + 107ul - *(generic64_t *)&_var_1606 + *(generic64_t *)&_var_1784 == 9223372036854775807ul ? 128u : 0u)) {
                                                _var_1438 = *(generic64_t *)&_var_1606 + 18446744073709551615ul - *(generic64_t *)&_var_1784;
                                              } else {
                                                goto _label_12;
                                              }
                                            }
                                          }
                                        } else {
                                          _var_1442 = 0ul;
                                          if ((generic64_t)((uint64_t)(uint32_t)*(generic32_t *)&_var_1781 % 10ul) == 0ul) {
                                            _var_1443 = 10ul;
                                            _var_1444 = 0ul;
                                          _label_13:
                                            _var_1443 = _var_1443 * 10ul & 4294967292ul;
                                            *(generic64_t *)&_var_1782 = _var_1444 + 1ul;
                                            _var_1444 = *(generic64_t *)&_var_1782 & 4294967295ul;
                                            if ((generic64_t)((uint64_t)(uint32_t)*(generic32_t *)&_var_1781 % (uint64_t)_var_1443) == 0ul) {
                                              goto _label_13;
                                            } else {
                                              _var_1442 = (generic64_t)((int64_t)(*(generic64_t *)&_var_1782 << 32ul) >> 32l);
                                            }
                                          } else {
                                          }
                                        }
                                      } else {
                                      }
                                    } else {
                                    }
                                  } else {
                                  }
                                }
                              } else {
                              }
                            } else {
                              if (((generic64_t)((uint64_t)*(generic64_t *)&_var_1767 / (uint64_t)_var_1500) & 1ul) == 0ul) {
                                if ((generic32_t)_var_1500 == 1000000000u ? (generic8_t)((uint64_t)*(generic64_t *)&_var_1763 < (uint64_t)(*(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul)) + 18446744073709486084ul)) : 0u) {
                                  if ((((generic8_t *)(*(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul))))[18446744073709486080ul] & 1u) == 0u) {
                                    helper_flds_ST0_wrapper((void *)0ul, *(generic32_t *)4215980ul, *(generic32_t *)&_var_1740, _var_1739, 0u, 0u, (void *)&_var_223, (void *)&_var_224, (void *)&_var_225, (void *)&_var_226, (void *)&_var_227, (void *)&_var_228, (void *)&_var_229, (void *)&_var_230, (void *)&_var_231, (void *)&_var_232, (void *)&_var_233, (void *)&_var_234, (void *)&_var_235, (void *)&_var_236, (void *)&_var_237, (void *)&_var_238, (void *)&_var_239, (void *)&_var_240, (void *)&_var_241, (void *)&_var_242, (void *)&_var_243, (void *)&_var_244, (void *)&_var_245, (void *)&_var_246, (void *)&_var_247, (void *)&_var_248);
                                    _var_1499 = _var_248;
                                    *(generic32_t **)&_var_1498 = &_var_223;
                                  } else {
                                    helper_fldt_ST0_wrapper((void *)0ul, 4216016ul, *(generic32_t *)&_var_1740, (void *)&_var_249, (void *)&_var_250, (void *)&_var_251, (void *)&_var_252, (void *)&_var_253, (void *)&_var_254, (void *)&_var_255, (void *)&_var_256, (void *)&_var_257, (void *)&_var_258, (void *)&_var_259, (void *)&_var_260, (void *)&_var_261, (void *)&_var_262, (void *)&_var_263, (void *)&_var_264, (void *)&_var_265, (void *)&_var_266, (void *)&_var_267, (void *)&_var_268, (void *)&_var_269, (void *)&_var_270, (void *)&_var_271, (void *)&_var_272, (void *)&_var_273);
                                    *(generic32_t **)&_var_1498 = &_var_249;
                                    _var_1499 = _var_1739;
                                  }
                                } else {
                                }
                              } else {
                              }
                              if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500 < (uint64_t)(uint32_t)((int32_t)(generic32_t)_var_1500 >> 1))) {
                                helper_flds_ST0_wrapper((void *)0ul, *(generic32_t *)4215988ul, *(generic32_t *)_var_1498, _var_1499, 0u, 0u, (void *)&_var_197, (void *)&_var_198, (void *)&_var_199, (void *)&_var_200, (void *)&_var_201, (void *)&_var_202, (void *)&_var_203, (void *)&_var_204, (void *)&_var_205, (void *)&_var_206, (void *)&_var_207, (void *)&_var_208, (void *)&_var_209, (void *)&_var_210, (void *)&_var_211, (void *)&_var_212, (void *)&_var_213, (void *)&_var_214, (void *)&_var_215, (void *)&_var_216, (void *)&_var_217, (void *)&_var_218, (void *)&_var_219, (void *)&_var_220, (void *)&_var_221, (void *)&_var_222);
                                _var_1479 = _var_197;
                                _var_1480 = _var_206;
                                _var_1481 = _var_207;
                                _var_1482 = _var_208;
                                _var_1483 = _var_209;
                                _var_1484 = _var_210;
                                _var_1485 = _var_211;
                                _var_1486 = _var_212;
                                _var_1487 = _var_213;
                                _var_1488 = _var_214;
                                _var_1489 = _var_215;
                                _var_1490 = _var_216;
                                _var_1491 = _var_217;
                                _var_1492 = _var_218;
                                _var_1493 = _var_219;
                                _var_1494 = _var_220;
                                _var_1495 = _var_221;
                                _var_1496 = _var_222;
                              } else {
                                *(generic64_t *)&_var_1604 = lshift((generic64_t)(uint64_t)(uint32_t)((generic32_t)((int32_t)(generic32_t)_var_1500 >> 1) - (generic32_t)(generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500)), 4294967272u);
                                *(generic64_t *)&_var_1605 = lshift((generic64_t)(uint64_t)(uint32_t)(((generic32_t)((int32_t)(generic32_t)_var_1500 >> 1) ^ (generic32_t)(generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500)) & ((generic32_t)((int32_t)(generic32_t)_var_1500 >> 1) ^ (generic32_t)((int32_t)(generic32_t)_var_1500 >> 1) - (generic32_t)(generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500))), 4294967276u);
                                _var_1497 = *(generic32_t *)_var_1498;
                                if ((generic32_t)((int32_t)(generic32_t)_var_1500 >> 1) == (generic32_t)(generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500)) {
                                  helper_fpush_wrapper((void *)0ul, *(generic32_t *)_var_1498, (void *)&_var_162, (void *)&_var_163, (void *)&_var_164, (void *)&_var_165, (void *)&_var_166, (void *)&_var_167, (void *)&_var_168, (void *)&_var_169, (void *)&_var_170);
                                  _var_1479 = _var_162;
                                  helper_fld1_ST0_wrapper((void *)0ul, _var_1479, (void *)&_var_146, (void *)&_var_147, (void *)&_var_148, (void *)&_var_149, (void *)&_var_150, (void *)&_var_151, (void *)&_var_152, (void *)&_var_153, (void *)&_var_154, (void *)&_var_155, (void *)&_var_156, (void *)&_var_157, (void *)&_var_158, (void *)&_var_159, (void *)&_var_160, (void *)&_var_161);
                                  _var_1480 = _var_146;
                                  _var_1481 = _var_147;
                                  _var_1482 = _var_148;
                                  _var_1483 = _var_149;
                                  _var_1484 = _var_150;
                                  _var_1485 = _var_151;
                                  _var_1486 = _var_152;
                                  _var_1487 = _var_153;
                                  _var_1488 = _var_154;
                                  _var_1489 = _var_155;
                                  _var_1490 = _var_156;
                                  _var_1491 = _var_157;
                                  _var_1492 = _var_158;
                                  _var_1493 = _var_159;
                                  _var_1494 = _var_160;
                                  _var_1495 = _var_161;
                                  _var_1496 = _var_1499;
                                  if (*(generic64_t *)&_var_1762 == *(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul)) + 18446744073709486088ul) {
                                  } else {
                                    helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_162, _var_146, _var_147, _var_148, _var_149, _var_150, _var_151, _var_152, _var_153, _var_154, _var_155, _var_156, _var_157, _var_158, _var_159, _var_160, _var_161, (void *)&_var_73, (void *)&_var_74, (void *)&_var_75, (void *)&_var_76, (void *)&_var_77, (void *)&_var_78, (void *)&_var_79, (void *)&_var_80, (void *)&_var_81, (void *)&_var_82, (void *)&_var_83, (void *)&_var_84, (void *)&_var_85, (void *)&_var_86, (void *)&_var_87, (void *)&_var_88);
                                    helper_fpop_wrapper((void *)0ul, _var_162, (void *)&_var_64, (void *)&_var_65, (void *)&_var_66, (void *)&_var_67, (void *)&_var_68, (void *)&_var_69, (void *)&_var_70, (void *)&_var_71, (void *)&_var_72);
                                    _var_1497 = _var_64;
                                    helper_flds_ST0_wrapper((void *)0ul, *(generic32_t *)4215984ul, _var_1497, _var_1499, 0u, 0u, (void *)&_var_171, (void *)&_var_172, (void *)&_var_173, (void *)&_var_174, (void *)&_var_175, (void *)&_var_176, (void *)&_var_177, (void *)&_var_178, (void *)&_var_179, (void *)&_var_180, (void *)&_var_181, (void *)&_var_182, (void *)&_var_183, (void *)&_var_184, (void *)&_var_185, (void *)&_var_186, (void *)&_var_187, (void *)&_var_188, (void *)&_var_189, (void *)&_var_190, (void *)&_var_191, (void *)&_var_192, (void *)&_var_193, (void *)&_var_194, (void *)&_var_195, (void *)&_var_196);
                                    _var_1479 = _var_171;
                                    _var_1480 = _var_180;
                                    _var_1481 = _var_181;
                                    _var_1482 = _var_182;
                                    _var_1483 = _var_183;
                                    _var_1484 = _var_184;
                                    _var_1485 = _var_185;
                                    _var_1486 = _var_186;
                                    _var_1487 = _var_187;
                                    _var_1488 = _var_188;
                                    _var_1489 = _var_189;
                                    _var_1490 = _var_190;
                                    _var_1491 = _var_191;
                                    _var_1492 = _var_192;
                                    _var_1493 = _var_193;
                                    _var_1494 = _var_194;
                                    _var_1495 = _var_195;
                                    _var_1496 = _var_196;
                                  }
                                } else {
                                }
                              }
                              _var_1463 = _var_1480;
                              _var_1464 = _var_1481;
                              _var_1465 = _var_1482;
                              _var_1466 = _var_1483;
                              _var_1467 = _var_1484;
                              _var_1468 = _var_1485;
                              _var_1469 = _var_1486;
                              _var_1470 = _var_1487;
                              _var_1471 = _var_1488;
                              _var_1472 = _var_1489;
                              _var_1473 = _var_1490;
                              _var_1474 = _var_1491;
                              _var_1475 = _var_1492;
                              _var_1476 = _var_1493;
                              _var_1477 = _var_1494;
                              _var_1478 = _var_1495;
                              if (((generic32_t *)&_var_1346)[12ul] == 0u) {
                              } else {
                                *(int8_t *)&_var_1768 = *(generic8_t *)*(generic64_t *)&_var_1670 == 45u;
                                _var_1463 = _var_1480;
                                _var_1464 = _var_1481;
                                _var_1465 = _var_1482;
                                _var_1466 = _var_1483;
                                _var_1467 = _var_1484;
                                _var_1468 = _var_1485;
                                _var_1469 = _var_1486;
                                _var_1470 = _var_1487;
                                _var_1471 = _var_1488;
                                _var_1472 = _var_1489;
                                _var_1473 = _var_1490;
                                _var_1474 = _var_1491;
                                _var_1475 = _var_1492;
                                _var_1476 = _var_1493;
                                _var_1477 = _var_1494;
                                _var_1478 = _var_1495;
                                if (_var_1768) {
                                  helper_fxchg_ST0_STN_wrapper((void *)0ul, 1u, _var_1479, _var_1480, _var_1481, _var_1482, _var_1483, _var_1484, _var_1485, _var_1486, _var_1487, _var_1488, _var_1489, _var_1490, _var_1491, _var_1492, _var_1493, _var_1494, _var_1495, (void *)&_var_48, (void *)&_var_49, (void *)&_var_50, (void *)&_var_51, (void *)&_var_52, (void *)&_var_53, (void *)&_var_54, (void *)&_var_55, (void *)&_var_56, (void *)&_var_57, (void *)&_var_58, (void *)&_var_59, (void *)&_var_60, (void *)&_var_61, (void *)&_var_62, (void *)&_var_63);
                                  helper_fchs_ST0_wrapper((void *)0ul, _var_1479, _var_48, _var_49, _var_50, _var_51, _var_52, _var_53, _var_54, _var_55, _var_56, _var_57, _var_58, _var_59, _var_60, _var_61, _var_62, _var_63, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42, (void *)&_var_43, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47);
                                  helper_fxchg_ST0_STN_wrapper((void *)0ul, 1u, _var_1479, _var_32, _var_33, _var_34, _var_35, _var_36, _var_37, _var_38, _var_39, _var_40, _var_41, _var_42, _var_43, _var_44, _var_45, _var_46, _var_47, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31);
                                  helper_fchs_ST0_wrapper((void *)0ul, _var_1479, _var_16, _var_17, _var_18, _var_19, _var_20, _var_21, _var_22, _var_23, _var_24, _var_25, _var_26, _var_27, _var_28, _var_29, _var_30, _var_31, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15);
                                  _var_1463 = _var_0;
                                  _var_1464 = _var_1;
                                  _var_1465 = _var_2;
                                  _var_1466 = _var_3;
                                  _var_1467 = _var_4;
                                  _var_1468 = _var_5;
                                  _var_1469 = _var_6;
                                  _var_1470 = _var_7;
                                  _var_1471 = _var_8;
                                  _var_1472 = _var_9;
                                  _var_1473 = _var_10;
                                  _var_1474 = _var_11;
                                  _var_1475 = _var_12;
                                  _var_1476 = _var_13;
                                  _var_1477 = _var_14;
                                  _var_1478 = _var_15;
                                } else {
                                }
                              }
                              helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, _var_1479, _var_1463, _var_1464, _var_1465, _var_1466, _var_1467, _var_1468, _var_1469, _var_1470, _var_1471, _var_1472, _var_1473, _var_1474, _var_1475, _var_1476, _var_1477, _var_1478, (void *)&_var_144, (void *)&_var_145);
                              helper_fadd_ST0_FT0_wrapper((void *)0ul, _var_1479, _var_1463, _var_1464, _var_1465, _var_1466, _var_1467, _var_1468, _var_1469, _var_1470, _var_1471, _var_1472, _var_1473, _var_1474, _var_1475, _var_1476, _var_1477, _var_1478, 0u, _var_1736, _var_1496, _var_1737, 0u, 0u, _var_144, _var_145, (void *)&_var_127, (void *)&_var_128, (void *)&_var_129, (void *)&_var_130, (void *)&_var_131, (void *)&_var_132, (void *)&_var_133, (void *)&_var_134, (void *)&_var_135, (void *)&_var_136, (void *)&_var_137, (void *)&_var_138, (void *)&_var_139, (void *)&_var_140, (void *)&_var_141, (void *)&_var_142, (void *)&_var_143);
                              helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, _var_1479, _var_127, _var_128, _var_129, _var_130, _var_131, _var_132, _var_133, _var_134, _var_135, _var_136, _var_137, _var_138, _var_139, _var_140, _var_141, _var_142, (void *)&_var_125, (void *)&_var_126);
                              helper_fucomi_ST0_FT0_wrapper((void *)0ul, *(generic64_t *)&_var_1767 - (generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500), 16u, (generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500), 0ul, _var_1479, _var_127, _var_128, _var_129, _var_130, _var_131, _var_132, _var_133, _var_134, _var_135, _var_136, _var_137, _var_138, _var_139, _var_140, _var_141, _var_142, _var_143, _var_125, _var_126, (void *)&_var_123, (void *)&_var_124);
                              helper_fpop_wrapper((void *)0ul, _var_1479, (void *)&_var_114, (void *)&_var_115, (void *)&_var_116, (void *)&_var_117, (void *)&_var_118, (void *)&_var_119, (void *)&_var_120, (void *)&_var_121, (void *)&_var_122);
                              helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_114, _var_127, _var_128, _var_129, _var_130, _var_131, _var_132, _var_133, _var_134, _var_135, _var_136, _var_137, _var_138, _var_139, _var_140, _var_141, _var_142, (void *)&_var_98, (void *)&_var_99, (void *)&_var_100, (void *)&_var_101, (void *)&_var_102, (void *)&_var_103, (void *)&_var_104, (void *)&_var_105, (void *)&_var_106, (void *)&_var_107, (void *)&_var_108, (void *)&_var_109, (void *)&_var_110, (void *)&_var_111, (void *)&_var_112, (void *)&_var_113);
                              helper_fpop_wrapper((void *)0ul, _var_114, (void *)&_var_89, (void *)&_var_90, (void *)&_var_91, (void *)&_var_92, (void *)&_var_93, (void *)&_var_94, (void *)&_var_95, (void *)&_var_96, (void *)&_var_97);
                              if ((_var_123 & 68ul) == 64ul) {
                                ((generic32_t *)(*(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul))))[4611686018427371521ul] = (generic32_t)(*(generic64_t *)&_var_1767 - (generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500));
                                _var_1452 = *(generic64_t *)&_var_1763;
                                _var_1453 = *(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul)) + 18446744073709486084ul;
                                _var_1454 = *(generic64_t *)&_var_1765;
                              } else {
                                ((generic32_t *)(*(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul))))[4611686018427371521ul] = (generic32_t)_var_1500 + (generic32_t)(*(generic64_t *)&_var_1767 - (generic64_t)((uint64_t)*(generic64_t *)&_var_1767 % (uint64_t)_var_1500));
                                _var_1457 = *(generic64_t *)&_var_1763;
                                _var_1458 = *(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul)) + 18446744073709486084ul;
                                if ((generic8_t)((uint32_t)((generic32_t *)(*(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul))))[4611686018427371521ul] > 999999999u)) {
                                  _var_1460 = 0ul;
                                  _var_1461 = *(generic64_t *)&_var_1766 + ((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul)) + 18446744073709486084ul;
                                  _var_1462 = *(generic64_t *)&_var_1763;
                                _label_14:
                                  *(generic64_t *)&_var_1769 = _var_1460;
                                  *(generic64_t *)&_var_1771 = _var_1462;
                                  _var_1459 = *(generic64_t *)&_var_1771;
                                  *(generic64_t *)&_var_1772 = _var_1459;
                                  *(generic64_t *)&_var_1770 = _var_1461 + 18446744073709551612ul;
                                  *(generic32_t *)_var_1461 = 0u;
                                  if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1772 > (uint64_t)(((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + (generic64_t)&_var_1346 + *(generic64_t *)&_var_1766 + 18446744073709493608ul - (*(generic64_t *)&_var_1769 << 2ul)))) {
                                    _var_1459 = *(generic64_t *)&_var_1771 + 18446744073709551612ul;
                                    *(generic32_t *)_var_1459 = 0u;
                                  } else {
                                  }
                                  *(generic64_t *)&_var_1773 = _var_1459;
                                  *(generic32_t *)(((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + (generic64_t)&_var_1346 + *(generic64_t *)&_var_1766 + 18446744073709493608ul - (*(generic64_t *)&_var_1769 << 2ul)) = *(generic32_t *)(((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + (generic64_t)&_var_1346 + *(generic64_t *)&_var_1766 + 18446744073709493608ul - (*(generic64_t *)&_var_1769 << 2ul)) + 1u;
                                  _var_1460 = *(generic64_t *)&_var_1769 + 1ul;
                                  _var_1461 = *(generic64_t *)&_var_1770;
                                  if ((generic8_t)((uint32_t)*(generic32_t *)(((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + (generic64_t)&_var_1346 + *(generic64_t *)&_var_1766 + 18446744073709493608ul - (*(generic64_t *)&_var_1769 << 2ul)) > 999999999u)) {
                                    goto _label_14;
                                  } else {
                                    _var_1457 = *(generic64_t *)&_var_1773;
                                    _var_1458 = ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) + (generic64_t)&_var_1346 + *(generic64_t *)&_var_1766 + 18446744073709493608ul - (*(generic64_t *)&_var_1769 << 2ul);
                                    _var_1452 = _var_1457;
                                    _var_1453 = _var_1458;
                                    *(generic32_t *)&_var_1774 = *(generic32_t *)_var_1452;
                                    _var_1454 = (generic64_t)((uint64_t)((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) - _var_1452) >> 2ul) * 9ul & 4294967295ul;
                                    if ((generic8_t)((uint32_t)*(generic32_t *)&_var_1774 < 10u)) {
                                    } else {
                                      _var_1455 = (generic64_t)((uint64_t)((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) - _var_1452) >> 2ul) * 9ul & 4294967295ul;
                                      _var_1456 = 10u;
                                    _label_15:
                                      _var_1456 = _var_1456 * 10u;
                                      _var_1455 = _var_1455 + 1ul & 4294967295ul;
                                      if ((generic8_t)((uint32_t)(*(generic32_t *)&_var_1774 - _var_1456) > (uint32_t)(_var_1456 ^ 4294967295u))) {
                                        _var_1452 = _var_1457;
                                        _var_1453 = _var_1458;
                                        _var_1454 = _var_1455;
                                      } else {
                                        goto _label_15;
                                      }
                                    }
                                  }
                                } else {
                                }
                              }
                            }
                          } else {
                            _var_1501 = 10ul;
                            _var_1502 = (generic64_t)((int64_t)(((generic8_t)((int32_t)((generic32_t)((((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) - (*(generic32_t *)&_var_1728 == 102u ? 0ul : _var_1451) & 4294967295ul) + (generic64_t)(int64_t)(int8_t)((generic8_t)(*(generic32_t *)&_var_1728 == 103u) & (generic8_t)(((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) != 0ul))) + 147456u) > 4294967295) ? 0ul : 18446744069414584320ul) | (generic64_t)(uint64_t)(uint32_t)((generic32_t)((((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) - (*(generic32_t *)&_var_1728 == 102u ? 0ul : _var_1451) & 4294967295ul) + (generic64_t)(int64_t)(int8_t)((generic8_t)(*(generic32_t *)&_var_1728 == 103u) & (generic8_t)(((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) != 0ul))) + 147456u)) % 9l);
                          _label_16:
                            _var_1502 = _var_1502 + 1ul & 4294967295ul;
                            _var_1501 = _var_1501 * 10ul & 4294967292ul;
                            if (_var_1502 == 8ul) {
                              _var_1500 = _var_1501;
                            } else {
                              goto _label_16;
                            }
                          }
                        } else {
                        }
                      } else {
                        _var_1504 = (generic64_t)((uint64_t)((generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) - *(generic64_t *)&_var_1763) >> 2ul) * 9ul & 4294967295ul;
                        _var_1505 = 10u;
                      _label_17:
                        _var_1505 = _var_1505 * 10u;
                        _var_1504 = _var_1504 + 1ul & 4294967295ul;
                        if ((generic8_t)((uint32_t)(*(generic32_t *)&_var_1764 - _var_1505) > (uint32_t)(_var_1505 ^ 4294967295u))) {
                          _var_1503 = _var_1504;
                        } else {
                          goto _label_17;
                        }
                      }
                    } else {
                    }
                  } else {
                    _var_1518 = *(generic32_t *)&_var_1752;
                    _var_1519 = *(generic64_t *)&_var_1751;
                    _var_1520 = *(generic64_t *)&_var_1750;
                  _label_18:
                    *(generic32_t *)&_var_1754 = _var_1518;
                    *(generic64_t *)&_var_1755 = _var_1519;
                    *(generic64_t *)&_var_1756 = _var_1520;
                    _var_1515 = 1953125u;
                    _var_1516 = 512u;
                    _var_1517 = 9ul;
                    if ((generic8_t)((int32_t)*(generic32_t *)&_var_1754 < 4294967287)) {
                    } else {
                      _var_1517 = (generic64_t)(uint64_t)(uint32_t)(0u - *(generic32_t *)&_var_1754);
                      _var_1515 = (generic32_t)(1000000000u >> (uint32_t)(0u - *(generic32_t *)&_var_1754 & 31u));
                      _var_1516 = 1u << (0u - *(generic32_t *)&_var_1754 & 31u);
                    }
                    *(generic32_t *)&_var_1757 = _var_1516;
                    *(generic64_t *)&_var_1758 = _var_1517;
                    ((generic32_t *)&_var_1346)[14ul] = _var_1515;
                    _var_1511 = 0ul;
                    if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1755 < (uint64_t)*(generic64_t *)&_var_1756)) {
                      _var_1512 = 0ul;
                      _var_1513 = *(generic64_t *)&_var_1755;
                      _var_1514 = 0u;
                    _label_19:
                      *(generic64_t *)&_var_1759 = _var_1512;
                      *(generic32_t *)&_var_1760 = *(generic32_t *)_var_1513;
                      _var_1513 = _var_1513 + 4ul;
                      *(generic32_t *)_var_1513 = _var_1514 + (generic32_t)(generic64_t)((uint64_t)(uint32_t)*(generic32_t *)&_var_1760 >> (uint64_t)(*(generic64_t *)&_var_1758 & 31ul));
                      _var_1514 = (*(generic32_t *)&_var_1757 + 4294967295u & *(generic32_t *)&_var_1760) * ((generic32_t *)&_var_1346)[14ul];
                      _var_1512 = *(generic64_t *)&_var_1759 + 1ul;
                      if ((generic8_t)((uint64_t)(*(generic64_t *)&_var_1755 + 4ul + (*(generic64_t *)&_var_1759 << 2ul)) < (uint64_t)*(generic64_t *)&_var_1756)) {
                        goto _label_19;
                      } else {
                        _var_1511 = (generic64_t)(uint64_t)(uint32_t)_var_1514;
                        *(generic64_t *)&_var_1761 = *(generic32_t *)*(generic64_t *)&_var_1755 == 0u ? *(generic64_t *)&_var_1755 + 4ul : *(generic64_t *)&_var_1755;
                        _var_1510 = *(generic64_t *)&_var_1756;
                        if (_var_1511 == 0ul) {
                        } else {
                          *(generic32_t *)*(generic64_t *)&_var_1756 = (generic32_t)_var_1511;
                          _var_1510 = *(generic64_t *)&_var_1756 + 4ul;
                        }
                        _var_1509 = _var_1510;
                        if ((generic8_t)((int64_t)(_var_1509 - (*(generic32_t *)&_var_1728 == 102u ? (generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) : *(generic64_t *)&_var_1761)) >> 2l > (int64_t)(((generic64_t)((int64_t)(((generic8_t)((int32_t)((generic32_t)((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) + 29u) > 4294967295) ? 0ul : 18446744069414584320ul) | (generic64_t)(uint64_t)(uint32_t)((generic32_t)((generic8_t)(p > 4294967295) ? (generic64_t)(uint64_t)(uint32_t)p : 6ul) + 29u)) / 9l) << 32ul) + 4294967296ul) >> 32l)) {
                          _var_1509 = ((generic64_t *)&_var_1346)[8ul] + (*(generic32_t *)&_var_1728 == 102u ? (generic64_t)&_var_1346 + 7528ul + ((generic8_t)((int32_t)*(generic32_t *)&_var_1729 < 0) ? 18446744073709544220ul : 18446744073709551300ul) : *(generic64_t *)&_var_1761);
                        } else {
                        }
                        _var_1518 = *(generic32_t *)&_var_1754 + (generic32_t)*(generic64_t *)&_var_1758;
                        _var_1519 = *(generic64_t *)&_var_1761;
                        if ((generic8_t)((int32_t)_var_1518 > 4294967295)) {
                          _var_1506 = _var_1509;
                          _var_1507 = *(generic64_t *)&_var_1761;
                          _var_1508 = *(generic32_t *)&_var_1754 + (generic32_t)*(generic64_t *)&_var_1758;
                        } else {
                          goto _label_18;
                        }
                      }
                    } else {
                    }
                  }
                }
              }
            } else {
            }
          } else {
            _var_1528 = 0ul;
            _var_1529 = *(generic64_t *)&_var_1744 + 18446744073709551612ul;
            _var_1530 = 0ul;
          _label_20:
            *(generic64_t *)&_var_1745 = _var_1528;
            *(generic64_t *)&_var_1746 = _var_1529;
            *(generic64_t *)&_var_1747 = (_var_1530 & 4294967295ul) + ((generic64_t)(uint64_t)(uint32_t)*(generic32_t *)*(generic64_t *)&_var_1746 << (*(generic64_t *)&_var_1741 & 63ul));
            _var_1530 = (generic64_t)((uint64_t)*(generic64_t *)&_var_1747 / 1000000000ul);
            *(generic32_t *)*(generic64_t *)&_var_1746 = (generic32_t)(generic64_t)((uint64_t)*(generic64_t *)&_var_1747 % 1000000000ul);
            _var_1529 = *(generic64_t *)&_var_1746 + 18446744073709551612ul;
            _var_1528 = *(generic64_t *)&_var_1745 + 1ul;
            if ((generic8_t)((uint64_t)*(generic64_t *)&_var_1743 > (uint64_t)(*(generic64_t *)&_var_1744 + 18446744073709551608ul - (*(generic64_t *)&_var_1745 << 2ul)))) {
              _var_1527 = (generic64_t)((uint64_t)*(generic64_t *)&_var_1747 / 1000000000ul);
            } else {
              goto _label_20;
            }
          }
        } else {
        }
      } else {
        goto _label_2;
      }
    }
  } else {
    _var_1708 = (generic8_t)((generic32_t *)&_var_1346)[6ul] & 32u;
    helper_fmov_FT0_STN_wrapper((void *)0ul, 0u, *(generic32_t *)&_var_1690, *(generic64_t *)&_var_1691, *(generic16_t *)&_var_1692, *(generic64_t *)&_var_1693, *(generic16_t *)&_var_1694, *(generic64_t *)&_var_1695, *(generic16_t *)&_var_1696, *(generic64_t *)&_var_1697, *(generic16_t *)&_var_1698, *(generic64_t *)&_var_1699, *(generic16_t *)&_var_1700, *(generic64_t *)&_var_1701, *(generic16_t *)&_var_1702, *(generic64_t *)&_var_1703, *(generic16_t *)&_var_1704, *(generic64_t *)&_var_1705, *(generic16_t *)&_var_1706, (void *)&_var_1158, (void *)&_var_1159);
    helper_fucomi_ST0_FT0_wrapper((void *)0ul, (generic64_t)(uint64_t)(uint32_t)(((generic32_t *)&_var_1346)[6ul] & 32u), 24u, *(generic64_t *)&_var_1707, 0ul, *(generic32_t *)&_var_1690, *(generic64_t *)&_var_1691, *(generic16_t *)&_var_1692, *(generic64_t *)&_var_1693, *(generic16_t *)&_var_1694, *(generic64_t *)&_var_1695, *(generic16_t *)&_var_1696, *(generic64_t *)&_var_1697, *(generic16_t *)&_var_1698, *(generic64_t *)&_var_1699, *(generic16_t *)&_var_1700, *(generic64_t *)&_var_1701, *(generic16_t *)&_var_1702, *(generic64_t *)&_var_1703, *(generic16_t *)&_var_1704, *(generic64_t *)&_var_1705, *(generic16_t *)&_var_1706, 0u, _var_1158, _var_1159, (void *)&_var_1156, (void *)&_var_1157);
    helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_1690, (void *)&_var_1147, (void *)&_var_1148, (void *)&_var_1149, (void *)&_var_1150, (void *)&_var_1151, (void *)&_var_1152, (void *)&_var_1153, (void *)&_var_1154, (void *)&_var_1155);
    if (((generic8_t)_var_1156 & 4u) == 0u) {
      _var_1573 = _var_1708 == 0u ? 4215943ul : 4215939ul;
      if (((generic8_t)_var_1156 & 64u) == 0u) {
        _var_1573 = _var_1708 == 0u ? 4215951ul : 4215947ul;
      } else {
      }
    } else {
    }
    _var_1350 = _var_1573;
    _var_1348 = ((generic32_t *)&_var_1346)[12ul] + 3u;
    pad(f, 32, (int32_t)((generic32_t *)&_var_1346)[4ul], (int32_t)_var_1348, (int32_t)(((generic32_t *)&_var_1346)[5ul] & 4294901759u));
    out(f, (const int8_t *)*(generic64_t *)&_var_1670, (size_t)(int64_t)(int32_t)((generic32_t *)&_var_1346)[12ul]);
    _var_1349 = 3ul;
  }
  return (int32_t)_var_1347;
}

_ABI(SystemV_x86_64)
int32_t unreserved___signbitl(float128_t x) {
  return (int32_t)((generic32_t)((uint32_t)((generic32_t *)_init_local_sp())[4ul] >> 15u) & 1u);
}

_ABI(SystemV_x86_64)
int32_t unreserved___fpclassifyl(float128_t x) {
  generic32_t _var_0;
  generic8_t _var_1[8];
  generic8_t _var_2[8];
  generic64_t _var_3 = _init_local_sp();
  *(generic64_t *)&_var_1 = ((generic64_t *)_var_3)[1ul];
  *(generic64_t *)&_var_2 = ((generic64_t *)_var_3)[2ul] & 32767ul;
  if ((*(generic64_t *)&_var_2 | (generic64_t)((uint64_t)*(generic64_t *)&_var_1 >> 63ul)) == 0ul) {
    _var_0 = *(generic64_t *)&_var_1 == 0ul ? 2u : 3u;
  } else {
    _var_0 = 0u;
    if ((generic8_t)((int64_t)*(generic64_t *)&_var_1 > -1l)) {
    } else {
      _var_0 = 4u;
      if (*(generic64_t *)&_var_2 == 32767ul) {
        _var_0 = (generic32_t)(uint32_t)(uint8_t)((*(generic64_t *)&_var_1 & 9223372036854775807ul) == 0ul);
      } else {
      }
    }
  }
  return (int32_t)_var_0;
}

_ABI(SystemV_x86_64)
float128_t frexpl(float128_t x, int32_t *e) {
  struct _PACKED struct_570 {
    generic64_t offset_0;
    generic64_t offset_8;
    uint8_t padding_at_16[8];
    generic16_t offset_24;
    uint8_t padding_at_26[14];
  };
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
  generic8_t _var_285[40];
  generic64_t _var_286;
  generic32_t _var_287;
  generic64_t _var_288;
  generic8_t _var_289[16];
  generic8_t _var_290[8];
  generic8_t _var_291[4];
  generic8_t _var_292[8];
  generic8_t _var_293[4];
  generic8_t _var_294[4];
  generic8_t _var_295[8];
  generic8_t _var_296[2];
  generic8_t _var_297[8];
  generic8_t _var_298[2];
  generic8_t _var_299[8];
  generic8_t _var_300[2];
  generic8_t _var_301[8];
  generic8_t _var_302[2];
  generic8_t _var_303[8];
  generic8_t _var_304[2];
  generic8_t _var_305[8];
  generic8_t _var_306[2];
  generic8_t _var_307[8];
  generic8_t _var_308[2];
  generic8_t _var_309[8];
  generic8_t _var_310[2];
  generic8_t _var_311[4];
  helper_fldt_ST0_wrapper((void *)0ul, (generic64_t)&_var_285 + 48ul, 0u, (void *)&_var_260, (void *)&_var_261, (void *)&_var_262, (void *)&_var_263, (void *)&_var_264, (void *)&_var_265, (void *)&_var_266, (void *)&_var_267, (void *)&_var_268, (void *)&_var_269, (void *)&_var_270, (void *)&_var_271, (void *)&_var_272, (void *)&_var_273, (void *)&_var_274, (void *)&_var_275, (void *)&_var_276, (void *)&_var_277, (void *)&_var_278, (void *)&_var_279, (void *)&_var_280, (void *)&_var_281, (void *)&_var_282, (void *)&_var_283, (void *)&_var_284);
  *(generic64_t *)&_var_290 = ((generic64_t *)&_var_285)[7ul];
  helper_fpush_wrapper((void *)0ul, _var_260, (void *)&_var_251, (void *)&_var_252, (void *)&_var_253, (void *)&_var_254, (void *)&_var_255, (void *)&_var_256, (void *)&_var_257, (void *)&_var_258, (void *)&_var_259);
  helper_fmov_ST0_STN_wrapper((void *)0ul, 1u, _var_251, _var_269, _var_270, _var_271, _var_272, _var_273, _var_274, _var_275, _var_276, _var_277, _var_278, _var_279, _var_280, _var_281, _var_282, _var_283, _var_284, (void *)&_var_235, (void *)&_var_236, (void *)&_var_237, (void *)&_var_238, (void *)&_var_239, (void *)&_var_240, (void *)&_var_241, (void *)&_var_242, (void *)&_var_243, (void *)&_var_244, (void *)&_var_245, (void *)&_var_246, (void *)&_var_247, (void *)&_var_248, (void *)&_var_249, (void *)&_var_250);
  helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_285 + 16ul, _var_251, _var_235, _var_236, _var_237, _var_238, _var_239, _var_240, _var_241, _var_242, _var_243, _var_244, _var_245, _var_246, _var_247, _var_248, _var_249, _var_250);
  helper_fpop_wrapper((void *)0ul, _var_251, (void *)&_var_226, (void *)&_var_227, (void *)&_var_228, (void *)&_var_229, (void *)&_var_230, (void *)&_var_231, (void *)&_var_232, (void *)&_var_233, (void *)&_var_234);
  if ((*(generic64_t *)&_var_290 & 32767ul) == 0ul) {
    helper_fpush_wrapper((void *)0ul, _var_226, (void *)&_var_217, (void *)&_var_218, (void *)&_var_219, (void *)&_var_220, (void *)&_var_221, (void *)&_var_222, (void *)&_var_223, (void *)&_var_224, (void *)&_var_225);
    helper_fldz_ST0_wrapper((void *)0ul, _var_217, (void *)&_var_201, (void *)&_var_202, (void *)&_var_203, (void *)&_var_204, (void *)&_var_205, (void *)&_var_206, (void *)&_var_207, (void *)&_var_208, (void *)&_var_209, (void *)&_var_210, (void *)&_var_211, (void *)&_var_212, (void *)&_var_213, (void *)&_var_214, (void *)&_var_215, (void *)&_var_216);
    helper_fxchg_ST0_STN_wrapper((void *)0ul, 1u, _var_217, _var_201, _var_202, _var_203, _var_204, _var_205, _var_206, _var_207, _var_208, _var_209, _var_210, _var_211, _var_212, _var_213, _var_214, _var_215, _var_216, (void *)&_var_185, (void *)&_var_186, (void *)&_var_187, (void *)&_var_188, (void *)&_var_189, (void *)&_var_190, (void *)&_var_191, (void *)&_var_192, (void *)&_var_193, (void *)&_var_194, (void *)&_var_195, (void *)&_var_196, (void *)&_var_197, (void *)&_var_198, (void *)&_var_199, (void *)&_var_200);
    helper_fmov_FT0_STN_wrapper((void *)0ul, 1u, _var_217, _var_185, _var_186, _var_187, _var_188, _var_189, _var_190, _var_191, _var_192, _var_193, _var_194, _var_195, _var_196, _var_197, _var_198, _var_199, _var_200, (void *)&_var_183, (void *)&_var_184);
    helper_fucomi_ST0_FT0_wrapper((void *)0ul, 0ul, 23u, 16ul, 0ul, _var_217, _var_185, _var_186, _var_187, _var_188, _var_189, _var_190, _var_191, _var_192, _var_193, _var_194, _var_195, _var_196, _var_197, _var_198, _var_199, _var_200, 0u, _var_183, _var_184, (void *)&_var_181, (void *)&_var_182);
    helper_fpop_wrapper((void *)0ul, _var_217, (void *)&_var_172, (void *)&_var_173, (void *)&_var_174, (void *)&_var_175, (void *)&_var_176, (void *)&_var_177, (void *)&_var_178, (void *)&_var_179, (void *)&_var_180);
    helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_172, _var_185, _var_186, _var_187, _var_188, _var_189, _var_190, _var_191, _var_192, _var_193, _var_194, _var_195, _var_196, _var_197, _var_198, _var_199, _var_200, (void *)&_var_156, (void *)&_var_157, (void *)&_var_158, (void *)&_var_159, (void *)&_var_160, (void *)&_var_161, (void *)&_var_162, (void *)&_var_163, (void *)&_var_164, (void *)&_var_165, (void *)&_var_166, (void *)&_var_167, (void *)&_var_168, (void *)&_var_169, (void *)&_var_170, (void *)&_var_171);
    helper_fpop_wrapper((void *)0ul, _var_172, (void *)&_var_147, (void *)&_var_148, (void *)&_var_149, (void *)&_var_150, (void *)&_var_151, (void *)&_var_152, (void *)&_var_153, (void *)&_var_154, (void *)&_var_155);
    *(generic32_t *)&_var_293 = _var_147;
    if ((_var_181 & 68ul) == 64ul) {
      *(generic32_t *)e = 0u;
      _var_287 = *(generic32_t *)&_var_293;
      _var_288 = *(generic64_t *)&_var_290;
    } else {
      helper_fldt_ST0_wrapper((void *)0ul, (generic64_t)&_var_285 + 48ul, *(generic32_t *)&_var_293, (void *)&_var_97, (void *)&_var_98, (void *)&_var_99, (void *)&_var_100, (void *)&_var_101, (void *)&_var_102, (void *)&_var_103, (void *)&_var_104, (void *)&_var_105, (void *)&_var_106, (void *)&_var_107, (void *)&_var_108, (void *)&_var_109, (void *)&_var_110, (void *)&_var_111, (void *)&_var_112, (void *)&_var_113, (void *)&_var_114, (void *)&_var_115, (void *)&_var_116, (void *)&_var_117, (void *)&_var_118, (void *)&_var_119, (void *)&_var_120, (void *)&_var_121);
      *(generic32_t *)&_var_294 = _var_97;
      helper_flds_FT0_wrapper((void *)0ul, *(generic32_t *)4217976ul, _var_182, 0u, 0u, (void *)&_var_94, (void *)&_var_95, (void *)&_var_96);
      helper_fmul_ST0_FT0_wrapper((void *)0ul, *(generic32_t *)&_var_294, _var_106, _var_107, _var_108, _var_109, _var_110, _var_111, _var_112, _var_113, _var_114, _var_115, _var_116, _var_117, _var_118, _var_119, _var_120, _var_121, 0u, 0u, _var_94, 80u, 0u, 0u, _var_95, _var_96, (void *)&_var_77, (void *)&_var_78, (void *)&_var_79, (void *)&_var_80, (void *)&_var_81, (void *)&_var_82, (void *)&_var_83, (void *)&_var_84, (void *)&_var_85, (void *)&_var_86, (void *)&_var_87, (void *)&_var_88, (void *)&_var_89, (void *)&_var_90, (void *)&_var_91, (void *)&_var_92, (void *)&_var_93);
      *(generic64_t *)&_var_295 = _var_77;
      *(generic16_t *)&_var_296 = _var_78;
      *(generic64_t *)&_var_297 = _var_79;
      *(generic16_t *)&_var_298 = _var_80;
      *(generic64_t *)&_var_299 = _var_81;
      *(generic16_t *)&_var_300 = _var_82;
      *(generic64_t *)&_var_301 = _var_83;
      *(generic16_t *)&_var_302 = _var_84;
      *(generic64_t *)&_var_303 = _var_85;
      *(generic16_t *)&_var_304 = _var_86;
      *(generic64_t *)&_var_305 = _var_87;
      *(generic16_t *)&_var_306 = _var_88;
      *(generic64_t *)&_var_307 = _var_89;
      *(generic16_t *)&_var_308 = _var_90;
      *(generic64_t *)&_var_309 = _var_91;
      *(generic16_t *)&_var_310 = _var_92;
      ((generic64_t *)&_var_285)[1ul] = *(generic64_t *)&_var_290 & 4294934527ul;
      *(generic64_t *)&_var_285 = *(generic64_t *)&_var_290 & 4294934527ul;
      helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_285, *(generic32_t *)&_var_294, *(generic64_t *)&_var_295, *(generic16_t *)&_var_296, *(generic64_t *)&_var_297, *(generic16_t *)&_var_298, *(generic64_t *)&_var_299, *(generic16_t *)&_var_300, *(generic64_t *)&_var_301, *(generic16_t *)&_var_302, *(generic64_t *)&_var_303, *(generic16_t *)&_var_304, *(generic64_t *)&_var_305, *(generic16_t *)&_var_306, *(generic64_t *)&_var_307, *(generic16_t *)&_var_308, *(generic64_t *)&_var_309, *(generic16_t *)&_var_310);
      helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_294, (void *)&_var_68, (void *)&_var_69, (void *)&_var_70, (void *)&_var_71, (void *)&_var_72, (void *)&_var_73, (void *)&_var_74, (void *)&_var_75, (void *)&_var_76);
      *(generic32_t *)&_var_311 = _var_68;
      *(float128_t *)&_var_289 = frexpl((float128_t)((generic128_t)x & (generic128_t)18446744073709551615u), e);
      _var_288 = (generic64_t)*(generic128_t *)&_var_289;
      _var_286 = (generic64_t)(generic128_t)((uint128_t)*(generic128_t *)&_var_289 >> (uint128_t)64u);
      *(generic32_t *)e = *(generic32_t *)e + 4294967176u;
      helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_285 + 48ul, *(generic32_t *)&_var_311, *(generic64_t *)&_var_295, *(generic16_t *)&_var_296, *(generic64_t *)&_var_297, *(generic16_t *)&_var_298, *(generic64_t *)&_var_299, *(generic16_t *)&_var_300, *(generic64_t *)&_var_301, *(generic16_t *)&_var_302, *(generic64_t *)&_var_303, *(generic16_t *)&_var_304, *(generic64_t *)&_var_305, *(generic16_t *)&_var_306, *(generic64_t *)&_var_307, *(generic16_t *)&_var_308, *(generic64_t *)&_var_309, *(generic16_t *)&_var_310);
      helper_fpop_wrapper((void *)0ul, *(generic32_t *)&_var_311, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8);
      _var_287 = _var_0;
    }
  } else {
    helper_fmov_STN_ST0_wrapper((void *)0ul, 0u, _var_226, _var_235, _var_236, _var_237, _var_238, _var_239, _var_240, _var_241, _var_242, _var_243, _var_244, _var_245, _var_246, _var_247, _var_248, _var_249, _var_250, (void *)&_var_131, (void *)&_var_132, (void *)&_var_133, (void *)&_var_134, (void *)&_var_135, (void *)&_var_136, (void *)&_var_137, (void *)&_var_138, (void *)&_var_139, (void *)&_var_140, (void *)&_var_141, (void *)&_var_142, (void *)&_var_143, (void *)&_var_144, (void *)&_var_145, (void *)&_var_146);
    helper_fpop_wrapper((void *)0ul, _var_226, (void *)&_var_122, (void *)&_var_123, (void *)&_var_124, (void *)&_var_125, (void *)&_var_126, (void *)&_var_127, (void *)&_var_128, (void *)&_var_129, (void *)&_var_130);
    *(generic32_t *)&_var_291 = _var_122;
    _var_287 = *(generic32_t *)&_var_291;
    _var_286 = *(generic64_t *)&_var_290 & 32767ul;
    _var_288 = *(generic64_t *)&_var_290;
    if ((*(generic64_t *)&_var_290 & 32767ul) == 32767ul) {
    } else {
      _var_286 = (generic64_t)(uint64_t)(uint32_t)((generic32_t)(*(generic64_t *)&_var_290 & 32767ul) + 4294950914u);
      _var_288 = *(generic64_t *)&_var_290 & 18446744073709518848ul | 16382ul;
      *(generic64_t *)&_var_292 = _var_288;
      *(generic32_t *)e = (generic32_t)(*(generic64_t *)&_var_290 & 32767ul) + 4294950914u;
      ((generic16_t *)&_var_285)[12ul] = (generic16_t)*(generic64_t *)&_var_292;
      helper_fldt_ST0_wrapper((void *)0ul, (generic64_t)&_var_285 + 16ul, *(generic32_t *)&_var_291, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42);
      helper_fstt_ST0_wrapper((void *)0ul, (generic64_t)&_var_285 + 48ul, _var_18, _var_27, _var_28, _var_29, _var_30, _var_31, _var_32, _var_33, _var_34, _var_35, _var_36, _var_37, _var_38, _var_39, _var_40, _var_41, _var_42);
      helper_fpop_wrapper((void *)0ul, _var_18, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17);
      _var_287 = _var_9;
    }
  }
  helper_fldt_ST0_wrapper((void *)0ul, (generic64_t)&_var_285 + 48ul, _var_287, (void *)&_var_43, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47, (void *)&_var_48, (void *)&_var_49, (void *)&_var_50, (void *)&_var_51, (void *)&_var_52, (void *)&_var_53, (void *)&_var_54, (void *)&_var_55, (void *)&_var_56, (void *)&_var_57, (void *)&_var_58, (void *)&_var_59, (void *)&_var_60, (void *)&_var_61, (void *)&_var_62, (void *)&_var_63, (void *)&_var_64, (void *)&_var_65, (void *)&_var_66, (void *)&_var_67);
  return (float128_t)((generic128_t)(uint128_t)(uint64_t)_var_286 << (generic128_t)64u | (generic128_t)(uint128_t)(uint64_t)_var_288);
}

_ABI(SystemV_x86_64)
int8_t *strerror(int32_t e) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic32_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic64_t _var_6;
  generic64_t _var_7;
  generic64_t _var_8;
  generic64_t _var_9;
  generic64_t _var_10;
  generic64_t _var_11;
  generic64_t _var_12;
  generic8_t _var_13;
  generic32_t _var_14;
  generic64_t _var_15;
  generic8_t _var_16;
  generic8_t _var_17[8];
  generic8_t _var_18;
  generic8_t _var_19[8];
  generic8_t _var_20[8];
  generic8_t _var_21;
  _var_16 = *(generic8_t *)4217888ul;
  _var_13 = _var_16;
  _var_12 = 0ul;
  _var_14 = 0u;
  if (_var_13 == 0u) {
    _var_0 = 4216064ul;
    if ((_var_12 & 4294967295ul) == 0ul) {
    } else {
      _var_10 = (generic64_t)(uint64_t)(uint8_t)_var_13;
      _var_11 = (generic64_t)(uint64_t)(uint32_t)_var_14;
      _var_8 = _var_12;
      _var_9 = 4216064ul;
    _label_0:
      _var_7 = _var_9;
      _var_6 = _var_10;
      _var_4 = _var_11;
      _var_5 = _var_8 & 4294967295ul;
      *(generic64_t *)&_var_19 = _var_7 + 1ul;
      _var_2 = 0ul;
      _var_3 = 24u;
    _label_1:
      *(generic64_t *)&_var_20 = *(generic64_t *)&_var_19 + _var_2;
      _var_21 = *(generic8_t *)_var_7;
      _var_1 = 0ul;
      switch (_var_3) {
        case 30u:
          _var_1 = _var_4;
        case 20u:
          _var_1 = (generic64_t)(uint64_t)(uint8_t)((uint32_t)((generic32_t)_var_4 ^ 4294967295u) < (uint32_t)(generic32_t)_var_5);
        case 28u: {
        }
        case 18u:
          _var_1 = (generic64_t)(uint64_t)(uint8_t)((uint32_t)((generic32_t)_var_4 & 255u) > (uint32_t)((generic32_t)_var_5 + (generic32_t)_var_4 & 255u));
        case 16u:
          _var_1 = (generic64_t)(uint64_t)(uint8_t)((uint32_t)((generic32_t)_var_4 ^ 4294967295u) < (uint32_t)(generic32_t)_var_5);
        case 26u: {
        }
        default: {
        }
      }
      _var_7 = _var_7 + 1ul;
      _var_2 = _var_2 + 1ul;
      _var_3 = 22u;
      _var_5 = _var_6 & 18446744073709551360ul | (generic64_t)(uint64_t)(uint8_t)_var_21;
      _var_6 = _var_6 & 18446744073709551360ul | (generic64_t)(uint64_t)(uint8_t)_var_21;
      if (_var_21 == 0u) {
        _var_8 = (_var_8 & 4294967295ul) + 18446744073709551615ul;
        _var_9 = *(generic64_t *)&_var_20;
        _var_10 = _var_5;
        _var_11 = 0ul;
        if ((_var_8 & 4294967295ul) == 0ul) {
          _var_0 = *(generic64_t *)&_var_20;
        } else {
          goto _label_0;
        }
      } else {
        goto _label_1;
      }
    }
  } else {
    _var_12 = 0ul;
    _var_13 = _var_16;
    _var_14 = (generic32_t)e;
    if ((generic32_t)(uint32_t)(uint8_t)_var_16 == (generic32_t)e) {
    } else {
      _var_15 = 0ul;
    _label_2:
      *(generic64_t *)&_var_17 = _var_15;
      _var_18 = ((generic8_t *)*(generic64_t *)&_var_17)[4217889ul];
      if (_var_18 == 0u) {
        _var_12 = *(generic64_t *)&_var_17 + 1ul;
        _var_13 = _var_18;
        _var_14 = (generic32_t)e;
      } else {
        _var_15 = *(generic64_t *)&_var_17 + 1ul;
        if ((generic32_t)(uint32_t)(uint8_t)_var_18 == (generic32_t)e) {
        } else {
          goto _label_2;
        }
      }
    }
  }
  return (int8_t *)_var_0;
}

_ABI(SystemV_x86_64)
void *memchr(const void *src, int32_t c, size_t n) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic64_t _var_6;
  generic64_t _var_7;
  generic64_t _var_8;
  generic64_t _var_9;
  generic64_t _var_10;
  generic64_t _var_11;
  generic64_t _var_12;
  generic64_t _var_13;
  generic8_t _var_14;
  generic8_t _var_15;
  generic8_t _var_16[8];
  generic8_t _var_17[8];
  generic8_t _var_18[8];
  generic8_t _var_19;
  _var_9 = (generic64_t)n;
  _var_10 = (generic64_t)src;
  if (((generic64_t)src & 7ul) == 0ul) {
    _var_0 = 0ul;
    if (_var_9 == 0ul) {
    } else {
      *(int8_t *)&_var_15 = *(generic8_t *)_var_10 == (generic8_t)(generic32_t)c;
      _var_0 = _var_10;
      if (_var_15) {
      } else {
        _var_4 = _var_9;
        _var_5 = _var_10;
        if ((generic8_t)((uint64_t)_var_9 > 7ul)) {
          _var_6 = 0ul;
          _var_7 = _var_9;
          _var_8 = _var_10;
        _label_0:
          if (((*(generic64_t *)_var_8 ^ (generic64_t)(uint64_t)(uint32_t)((generic32_t)c & 255u) * 72340172838076673ul) + 18374403900871474943ul & (*(generic64_t *)_var_8 ^ (generic64_t)(uint64_t)(uint32_t)((generic32_t)c & 255u) * 72340172838076673ul ^ 18446744073709551615ul) & 9259542123273814144ul) == 0ul) {
            *(generic64_t *)&_var_16 = _var_6 << 3ul;
            _var_7 = _var_7 + 18446744073709551608ul;
            _var_8 = _var_8 + 8ul;
            _var_6 = _var_6 + 1ul;
            if ((generic8_t)((uint64_t)(_var_9 + 18446744073709551608ul - *(generic64_t *)&_var_16) > 7ul)) {
              goto _label_0;
            } else {
              _var_5 = _var_10 + 8ul + *(generic64_t *)&_var_16;
              _var_0 = 0ul;
              _var_4 = _var_9 + 18446744073709551608ul - *(generic64_t *)&_var_16;
              if (_var_9 + 18446744073709551608ul - *(generic64_t *)&_var_16 == 0ul) {
              } else {
                _var_3 = _var_5;
                *(generic64_t *)&_var_18 = _var_3 + _var_4;
                *(generic64_t *)&_var_17 = _var_3 + 1ul;
                _var_2 = 0ul;
              _label_1:
                _var_1 = _var_3;
                if (*(generic8_t *)_var_1 == (generic8_t)(generic32_t)c) {
                  _var_0 = _var_1;
                } else {
                  _var_3 = _var_3 + 1ul;
                  *(int8_t *)&_var_19 = *(generic64_t *)&_var_17 + _var_2 == *(generic64_t *)&_var_18;
                  _var_2 = _var_2 + 1ul;
                  _var_1 = 0ul;
                  if (_var_19) {
                  } else {
                    goto _label_1;
                  }
                }
              }
            }
          } else {
            _var_4 = _var_7;
            _var_5 = _var_8;
          }
        } else {
        }
      }
    }
  } else {
    _var_0 = 0ul;
    if ((generic64_t)n == 0ul) {
    } else {
      _var_12 = 0ul;
      _var_13 = (generic64_t)src;
    _label_2:
      _var_11 = _var_13;
      if (*(generic8_t *)_var_11 == (generic8_t)(generic32_t)c) {
        _var_0 = _var_11;
      } else if (((generic64_t)src + 1ul + _var_12 & 7ul) == 0ul) {
        _var_9 = (generic64_t)n + (generic64_t)src - ((generic64_t)src + 1ul + _var_12);
        _var_10 = (generic64_t)src + 1ul + _var_12;
      } else {
        _var_13 = _var_13 + 1ul;
        *(int8_t *)&_var_14 = (generic64_t)n + (generic64_t)src == (generic64_t)src + 1ul + _var_12;
        _var_12 = _var_12 + 1ul;
        _var_11 = 0ul;
        if (_var_14) {
        } else {
          goto _label_2;
        }
      }
    }
  }
  return (void *)_var_0;
}

_ABI(SystemV_x86_64)
int32_t wctomb(int8_t *s, wchar_t wc) {
  struct _PACKED struct_574 {
    uint8_t padding_at_0[8];
  };
  generic32_t _var_0;
  generic8_t _var_1[8];
  _var_0 = 0u;
  if ((generic64_t)s == 0ul) {
  } else {
    *(size_t *)&_var_1 = wcrtomb((typedef_332)s, wc, (typedef_350)0ul);
    _var_0 = (generic32_t)*(generic64_t *)&_var_1;
  }
  return (int32_t)_var_0;
}

_ABI(SystemV_x86_64)
size_t wcrtomb(typedef_332 s, wchar_t wc, typedef_350 st) {
  struct _PACKED struct_575 {
    uint8_t padding_at_0[8];
  };
  generic64_t _var_0;
  generic8_t _var_1[8];
  _var_0 = 1ul;
  if ((generic64_t)s == 0ul) {
  } else if ((generic8_t)((uint32_t)wc < 128u)) {
    *(generic8_t *)s = (generic8_t)(generic32_t)wc;
    _var_0 = 1ul;
  } else if (*(generic64_t *)((generic64_t *)*(generic64_t *)0ul)[32ul] == 0ul) {
    if (((generic32_t)wc & 4294967168u) == 57216u) {
      *(generic8_t *)s = (generic8_t)(generic32_t)wc;
      _var_0 = 1ul;
    } else {
      *(int32_t **)&_var_1 = unreserved___errno_location();
      *(generic32_t *)*(generic64_t *)&_var_1 = 84u;
      _var_0 = 18446744073709551615ul;
    }
  } else if ((generic8_t)((uint32_t)wc < 2048u)) {
    ((generic8_t *)s)[1ul] = (generic8_t)(generic32_t)wc & 63u | 128u;
    *(generic8_t *)s = (generic8_t)(generic64_t)((uint64_t)(uint32_t)wc >> 6ul) | 192u;
    _var_0 = 2ul;
  } else if (((generic64_t)(uint64_t)(uint32_t)wc + 4294909952ul & 4294959104ul) != 0ul ? (generic8_t)((uint32_t)wc > 55295u) : 0u) {
    if ((generic8_t)((uint32_t)((generic32_t)wc + 4294901760u) < 1048576u)) {
      *(generic8_t *)s = (generic8_t)(generic64_t)((uint64_t)(uint32_t)wc >> 18ul) | 240u;
      ((generic8_t *)s)[1ul] = (generic8_t)(generic64_t)((uint64_t)(uint32_t)wc >> 12ul) & 63u | 128u;
      ((generic8_t *)s)[3ul] = (generic8_t)(generic32_t)wc & 63u | 128u;
      ((generic8_t *)s)[2ul] = (generic8_t)(generic64_t)((uint64_t)(uint32_t)wc >> 6ul) & 63u | 128u;
      _var_0 = 4ul;
    } else {
    }
  } else {
    *(generic8_t *)s = (generic8_t)(generic64_t)((uint64_t)(uint32_t)wc >> 12ul) | 224u;
    ((generic8_t *)s)[2ul] = (generic8_t)(generic32_t)wc & 63u | 128u;
    ((generic8_t *)s)[1ul] = (generic8_t)(generic64_t)((uint64_t)(uint32_t)wc >> 6ul) & 63u | 128u;
    _var_0 = 3ul;
  }
  return (size_t)_var_0;
}

_ABI(SystemV_x86_64)
void unreserved___wait(typedef_315 *addr, typedef_315 *waiters, int32_t val, int32_t priv) {
  struct _PACKED struct_576 {
    generic64_t offset_0;
    generic64_t offset_8;
  };
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
  generic32_t _var_48;
  generic64_t _var_49;
  generic64_t _var_50;
  generic64_t _var_51;
  generic64_t _var_52;
  generic32_t _var_53;
  generic64_t _var_54;
  generic32_t _var_55;
  generic32_t _var_56;
  generic32_t _var_57;
  generic64_t _var_58;
  generic32_t _var_59;
  generic32_t _var_60;
  generic32_t _var_61;
  generic64_t _var_62;
  generic32_t _var_63;
  generic32_t _var_64;
  generic64_t _var_65;
  generic32_t _var_66;
  generic32_t _var_67;
  generic32_t _var_68;
  generic64_t _var_69;
  generic32_t _var_70;
  generic8_t _var_71;
  generic8_t _var_72[16];
  generic32_t _var_73;
  generic64_t _var_74;
  generic32_t _var_75;
  generic64_t _var_76;
  generic64_t _var_77;
  generic64_t _var_78;
  generic32_t _var_79;
  generic32_t _var_80;
  generic64_t _var_81;
  generic32_t _var_82;
  generic32_t _var_83;
  generic64_t _var_84;
  generic32_t _var_85;
  generic32_t _var_86;
  generic64_t _var_87;
  generic64_t _var_88;
  generic64_t _var_89;
  generic32_t _var_90;
  generic64_t _var_91;
  generic32_t _var_92;
  ((generic64_t *)&_var_72)[1ul] = 202ul;
  *(generic64_t *)&_var_72 = 202ul;
  if ((generic64_t)waiters == 0ul) {
    if (*(generic32_t *)addr == (generic32_t)val) {
      helper_pause_wrapper((void *)0ul, 2u, 4209783ul, 0ul, (generic64_t)waiters, /* undef */ (generic64_t){0}, 100ul, 202ul, (generic64_t)(uint64_t)(uint32_t)val, (generic32_t)priv == 0u ? 0ul : 128ul, (generic64_t)addr, (generic64_t)(uint64_t)(uint32_t)*(generic32_t *)addr, (generic64_t)waiters, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_48, (void *)&_var_49, (void *)&_var_50, (void *)&_var_51, (void *)&_var_52, (void *)&_var_53, (void *)&_var_54, (void *)&_var_55, (void *)&_var_56, (void *)&_var_57, (void *)&_var_58, (void *)&_var_59, (void *)&_var_60, (void *)&_var_61, (void *)&_var_62, (void *)&_var_63, (void *)&_var_64, (void *)&_var_65, (void *)&_var_66, (void *)&_var_67, (void *)&_var_68, (void *)&_var_69, (void *)&_var_70, (void *)&_var_71);
      _abort((void *)"A longjmp was taken");
      __builtin_unreachable();
    } else {
    }
  } else if (*(generic32_t *)waiters == 0u) {
  } else {
    helper_lock();
    *(generic32_t *)waiters = *(generic32_t *)waiters + 1u;
    helper_unlock();
    if (*(generic32_t *)addr == (generic32_t)val) {
      _var_83 = 4294967295u;
      _var_84 = 0ul;
      _var_85 = 0u;
      _var_86 = 65535u;
      _var_87 = 0ul;
      _var_88 = 0ul;
      _var_89 = 0ul;
      _var_90 = 4243635u;
      _var_91 = 514ul;
      _var_92 = 4294967295u;
    _label_0:
      helper_syscall_wrapper((void *)0ul, 2u, 4209837ul, 0ul, (generic64_t)waiters, (generic32_t)priv == 0u ? 0ul : 128ul, 202ul, 202ul, (generic64_t)(uint64_t)(uint32_t)val, (generic32_t)priv == 0u ? 0ul : 128ul, (generic64_t)addr, (generic64_t)(int64_t)val, (generic32_t)priv == 0u ? 0ul : 128ul, _var_92, _var_91, _var_90, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, _var_89, _var_88, _var_87, _var_86, 274877906944ul, 127u, 2147549185ul, 0ul, _var_85, _var_84, _var_83, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42, (void *)&_var_43, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47);
      _var_73 = _var_29;
      _var_74 = _var_30;
      _var_75 = _var_32;
      _var_76 = _var_34;
      _var_77 = _var_38;
      _var_78 = _var_41;
      _var_79 = _var_42;
      _var_80 = _var_43;
      _var_81 = _var_45;
      _var_82 = _var_46;
      if (_var_27 == 18446744073709551578ul) {
        helper_syscall_wrapper((void *)0ul, 2u, 4209850ul, 0ul, (generic64_t)waiters, (generic32_t)priv == 0u ? 0ul : 128ul, 202ul, 202ul, (generic64_t)(uint64_t)(uint32_t)val, (generic32_t)priv == 0u ? 0ul : 128ul, (generic64_t)addr, (generic64_t)(int64_t)val, 0ul, _var_29, _var_30, _var_32, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, _var_34, _var_38, _var_41, _var_42, 274877906944ul, 127u, 2147549185ul, 0ul, _var_43, _var_45, _var_46, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
        _var_73 = _var_5;
        _var_74 = _var_6;
        _var_75 = _var_8;
        _var_76 = _var_10;
        _var_77 = _var_14;
        _var_78 = _var_17;
        _var_79 = _var_18;
        _var_80 = _var_19;
        _var_81 = _var_21;
        _var_82 = _var_22;
      } else {
      }
      if (*(generic32_t *)addr == (generic32_t)val) {
        goto _label_0;
      } else {
        helper_lock();
        *(generic32_t *)waiters = *(generic32_t *)waiters + 4294967295u;
        helper_unlock();
      }
    } else {
    }
  }
  return;
}

_ABI(SystemV_x86_64)
int8_t *strerror_l(int32_t e, locale_t_ loc) {
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic32_t _var_3;
  generic64_t _var_4;
  generic64_t _var_5;
  generic64_t _var_6;
  generic64_t _var_7;
  generic64_t _var_8;
  generic64_t _var_9;
  generic64_t _var_10;
  generic64_t _var_11;
  generic64_t _var_12;
  generic8_t _var_13;
  generic32_t _var_14;
  generic64_t _var_15;
  generic8_t _var_16;
  generic8_t _var_17[8];
  generic8_t _var_18;
  generic8_t _var_19[8];
  generic8_t _var_20[8];
  generic8_t _var_21;
  _var_16 = *(generic8_t *)4217888ul;
  _var_13 = _var_16;
  _var_12 = 0ul;
  _var_14 = 0u;
  if (_var_13 == 0u) {
    _var_0 = 4216064ul;
    if ((_var_12 & 4294967295ul) == 0ul) {
    } else {
      _var_10 = (generic64_t)(uint64_t)(uint8_t)_var_13;
      _var_11 = (generic64_t)(uint64_t)(uint32_t)_var_14;
      _var_8 = _var_12;
      _var_9 = 4216064ul;
    _label_0:
      _var_7 = _var_9;
      _var_6 = _var_10;
      _var_4 = _var_11;
      _var_5 = _var_8 & 4294967295ul;
      *(generic64_t *)&_var_19 = _var_7 + 1ul;
      _var_2 = 0ul;
      _var_3 = 24u;
    _label_1:
      *(generic64_t *)&_var_20 = *(generic64_t *)&_var_19 + _var_2;
      _var_21 = *(generic8_t *)_var_7;
      _var_1 = 0ul;
      switch (_var_3) {
        case 30u:
          _var_1 = _var_4;
        case 20u:
          _var_1 = (generic64_t)(uint64_t)(uint8_t)((uint32_t)((generic32_t)_var_4 ^ 4294967295u) < (uint32_t)(generic32_t)_var_5);
        case 28u: {
        }
        case 18u:
          _var_1 = (generic64_t)(uint64_t)(uint8_t)((uint32_t)((generic32_t)_var_4 & 255u) > (uint32_t)((generic32_t)_var_5 + (generic32_t)_var_4 & 255u));
        case 16u:
          _var_1 = (generic64_t)(uint64_t)(uint8_t)((uint32_t)((generic32_t)_var_4 ^ 4294967295u) < (uint32_t)(generic32_t)_var_5);
        case 26u: {
        }
        default: {
        }
      }
      _var_7 = _var_7 + 1ul;
      _var_2 = _var_2 + 1ul;
      _var_3 = 22u;
      _var_5 = _var_6 & 18446744073709551360ul | (generic64_t)(uint64_t)(uint8_t)_var_21;
      _var_6 = _var_6 & 18446744073709551360ul | (generic64_t)(uint64_t)(uint8_t)_var_21;
      if (_var_21 == 0u) {
        _var_8 = (_var_8 & 4294967295ul) + 18446744073709551615ul;
        _var_9 = *(generic64_t *)&_var_20;
        _var_10 = _var_5;
        _var_11 = 0ul;
        if ((_var_8 & 4294967295ul) == 0ul) {
          _var_0 = *(generic64_t *)&_var_20;
        } else {
          goto _label_0;
        }
      } else {
        goto _label_1;
      }
    }
  } else {
    _var_12 = 0ul;
    _var_13 = _var_16;
    _var_14 = (generic32_t)e;
    if ((generic32_t)(uint32_t)(uint8_t)_var_16 == (generic32_t)e) {
    } else {
      _var_15 = 0ul;
    _label_2:
      *(generic64_t *)&_var_17 = _var_15;
      _var_18 = ((generic8_t *)*(generic64_t *)&_var_17)[4217889ul];
      if (_var_18 == 0u) {
        _var_12 = *(generic64_t *)&_var_17 + 1ul;
        _var_13 = _var_18;
        _var_14 = (generic32_t)e;
      } else {
        _var_15 = *(generic64_t *)&_var_17 + 1ul;
        if ((generic32_t)(uint32_t)(uint8_t)_var_18 == (generic32_t)e) {
        } else {
          goto _label_2;
        }
      }
    }
  }
  return (int8_t *)_var_0;
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
int32_t dummy___(int32_t fd) {
  return fd;
}

_ABI(SystemV_x86_64)
int32_t unreserved___stdio_close(FILE_ *f) {
  struct _PACKED struct_577 {
    uint8_t padding_at_0[8];
  };
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
  generic8_t _var_24[4];
  generic8_t _var_25[8];
  *(int32_t *)&_var_24 = dummy___((int32_t)((generic32_t *)f)[30ul]);
  helper_syscall_wrapper((void *)0ul, 2u, 4209267ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 3ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)(int32_t)*(generic32_t *)&_var_24, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
  *(int64_t *)&_var_25 = unreserved___syscall_ret((uint64_t)_var_3);
  return (int32_t)(generic32_t)*(generic64_t *)&_var_25;
}

_ABI(SystemV_x86_64)
int64_t unreserved___syscall_ret(uint64_t r) {
  struct _PACKED struct_578 {
    uint8_t padding_at_0[8];
    generic64_t offset_8;
    uint8_t padding_at_16[8];
  };
  generic8_t _var_0[24];
  generic64_t _var_1;
  generic8_t _var_2[8];
  _var_1 = (generic64_t)r;
  if ((generic8_t)(r > 18446744073709547520ul)) {
    ((generic64_t *)&_var_0)[1ul] = (generic64_t)r;
    *(int32_t **)&_var_2 = unreserved___errno_location();
    *(generic32_t *)*(generic64_t *)&_var_2 = 0u - (generic32_t)((generic64_t *)&_var_0)[1ul];
    _var_1 = 18446744073709551615ul;
  } else {
  }
  return (int64_t)_var_1;
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
  generic8_t _var_25[8];
  generic8_t _var_26[8];
  helper_syscall_wrapper((void *)0ul, 2u, 4209291ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 8ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)(int32_t)((generic32_t *)f)[30ul], (generic64_t)(int64_t)whence, (generic64_t)off, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
  _var_24 = _var_3;
  if ((generic8_t)((uint64_t)_var_24 > 18446744073709547520ul)) {
    *(void **)&_var_26 = (void *)(_var_4 + 18446744073709551600ul);
    *(generic64_t *)*(void **)&_var_26 = _var_3;
    *(int32_t **)&_var_25 = unreserved___errno_location();
    *(generic32_t *)*(generic64_t *)&_var_25 = 0u - (generic32_t)*(generic64_t *)*(void **)&_var_26;
    _var_24 = 18446744073709551615ul;
  } else {
  }
  return (off_t)_var_24;
}

_ABI(SystemV_x86_64)
size_t unreserved___stdout_write(FILE_ *f, const uint8_t *buf, size_t len) {
  struct _PACKED struct_579 {
    uint8_t padding_at_0[1];
  };
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
  generic8_t _var_24;
  generic8_t _var_25[8];
  ((generic64_t *)f)[9ul] = 4210190ul;
  if ((*(generic8_t *)f & 64u) == 0u) {
    helper_syscall_wrapper((void *)0ul, 2u, 4209346ul, (generic64_t)len, (generic64_t)f, (generic64_t)buf, 16ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)(int32_t)((generic32_t *)f)[30ul], (generic64_t)&_var_24 + 18446744073709551601ul, 21523ul, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
    if (_var_3 == 0ul) {
    } else {
      ((generic8_t *)f)[139ul] = 255u;
    }
  } else {
  }
  *(size_t *)&_var_25 = unreserved___stdio_write(f, buf, len);
  return (size_t)*(generic64_t *)&_var_25;
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
  };
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
  generic8_t _var_24[88];
  generic64_t _var_25;
  generic32_t _var_26;
  generic64_t _var_27;
  generic32_t _var_28;
  generic64_t _var_29;
  generic64_t _var_30;
  generic64_t _var_31;
  generic64_t _var_32;
  generic32_t _var_33;
  generic64_t _var_34;
  generic64_t _var_35;
  generic32_t _var_36;
  generic64_t _var_37;
  generic32_t _var_38;
  generic8_t _var_39[8];
  generic8_t _var_40[8];
  generic8_t _var_41[8];
  generic8_t _var_42[8];
  generic8_t _var_43[8];
  generic8_t _var_44[8];
  generic8_t _var_45[4];
  generic8_t _var_46[8];
  generic8_t _var_47[4];
  generic8_t _var_48[8];
  generic8_t _var_49[8];
  generic8_t _var_50[8];
  generic8_t _var_51[4];
  generic8_t _var_52[4];
  generic8_t _var_53[8];
  generic8_t _var_54[4];
  generic8_t _var_55[8];
  generic8_t _var_56[8];
  generic8_t _var_57[8];
  generic8_t _var_58[8];
  _var_29 = (generic64_t)&_var_24;
  *(generic64_t *)&_var_40 = _var_29;
  ((generic64_t *)*(generic64_t *)&_var_40)[8ul] = 20ul;
  *(generic64_t *)&_var_41 = ((generic64_t *)f)[7ul];
  *(generic64_t *)&_var_42 = ((generic64_t *)f)[5ul];
  ((generic64_t *)*(generic64_t *)&_var_40)[2ul] = (generic64_t)buf;
  *(generic64_t *)&_var_24 = *(generic64_t *)&_var_41;
  ((generic64_t *)*(generic64_t *)&_var_40)[3ul] = (generic64_t)len;
  ((generic64_t *)*(generic64_t *)&_var_40)[1ul] = *(generic64_t *)&_var_42 - *(generic64_t *)&_var_41;
  _var_35 = *(generic64_t *)&_var_42 - *(generic64_t *)&_var_41 + (generic64_t)len;
  _var_26 = 4294967295u;
  _var_27 = 514ul;
  _var_28 = 4243635u;
  _var_30 = 0ul;
  _var_31 = 0ul;
  _var_32 = 0ul;
  _var_33 = 65535u;
  _var_34 = 2ul;
  _var_36 = 0u;
  _var_37 = 0ul;
  _var_38 = 4294967295u;
_label_0:
  *(generic64_t *)&_var_43 = _var_29;
  *(generic64_t *)&_var_44 = _var_34;
  helper_syscall_wrapper((void *)0ul, 2u, 4210271ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 20ul, *(generic64_t *)&_var_43, (generic64_t)f, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)(int32_t)((generic32_t *)f)[30ul], (generic64_t)((int64_t)(*(generic64_t *)&_var_44 << 32ul) >> 32l), *(generic64_t *)&_var_43, _var_26, _var_27, _var_28, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, _var_30, _var_31, _var_32, _var_33, 274877906944ul, 127u, 2147549185ul, 0ul, _var_36, _var_37, _var_38, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
  *(generic32_t *)&_var_45 = _var_5;
  *(generic64_t *)&_var_46 = _var_6;
  *(generic32_t *)&_var_47 = _var_8;
  *(generic64_t *)&_var_48 = _var_10;
  *(generic64_t *)&_var_49 = _var_14;
  *(generic64_t *)&_var_50 = _var_17;
  *(generic32_t *)&_var_51 = _var_18;
  *(generic32_t *)&_var_52 = _var_19;
  *(generic64_t *)&_var_53 = _var_21;
  *(generic32_t *)&_var_54 = _var_22;
  *(int64_t *)&_var_39 = unreserved___syscall_ret((uint64_t)_var_3);
  if (_var_35 == *(generic64_t *)&_var_39) {
    *(generic64_t *)&_var_57 = ((generic64_t *)f)[11ul];
    *(generic64_t *)&_var_58 = ((generic64_t *)f)[12ul] + *(generic64_t *)&_var_57;
    ((generic64_t *)f)[7ul] = *(generic64_t *)&_var_57;
    ((generic64_t *)f)[4ul] = *(generic64_t *)&_var_58;
    ((generic64_t *)f)[5ul] = *(generic64_t *)&_var_57;
    _var_25 = (generic64_t)len;
  } else if ((generic8_t)((int64_t)*(generic64_t *)&_var_39 > -1l)) {
    *(generic64_t *)&_var_55 = ((generic64_t *)*(generic64_t *)&_var_43)[1ul];
    _var_35 = _var_35 - *(generic64_t *)&_var_39;
    _var_29 = (generic8_t)((uint64_t)*(generic64_t *)&_var_39 > (uint64_t)*(generic64_t *)&_var_55) ? *(generic64_t *)&_var_43 + 16ul : *(generic64_t *)&_var_43;
    *(generic64_t *)&_var_56 = _var_29;
    _var_34 = (generic8_t)((uint64_t)*(generic64_t *)&_var_39 > (uint64_t)*(generic64_t *)&_var_55) ? *(generic64_t *)&_var_44 + 4294967295ul & 4294967295ul : *(generic64_t *)&_var_44;
    *(generic64_t *)*(generic64_t *)&_var_56 = *(generic64_t *)*(generic64_t *)&_var_56 + (*(generic64_t *)&_var_39 - ((generic8_t)((uint64_t)*(generic64_t *)&_var_39 > (uint64_t)*(generic64_t *)&_var_55) ? *(generic64_t *)&_var_55 : 0ul));
    ((generic64_t *)*(generic64_t *)&_var_56)[1ul] = ((generic64_t *)*(generic64_t *)&_var_56)[1ul] - (*(generic64_t *)&_var_39 - ((generic8_t)((uint64_t)*(generic64_t *)&_var_39 > (uint64_t)*(generic64_t *)&_var_55) ? *(generic64_t *)&_var_55 : 0ul));
    _var_26 = *(generic32_t *)&_var_45;
    _var_27 = *(generic64_t *)&_var_46;
    _var_28 = *(generic32_t *)&_var_47;
    _var_30 = *(generic64_t *)&_var_48;
    _var_31 = *(generic64_t *)&_var_49;
    _var_32 = *(generic64_t *)&_var_50;
    _var_33 = *(generic32_t *)&_var_51;
    _var_36 = *(generic32_t *)&_var_52;
    _var_37 = *(generic64_t *)&_var_53;
    _var_38 = *(generic32_t *)&_var_54;
    goto _label_0;
  } else {
    *(generic32_t *)f = *(generic32_t *)f | 32u;
    ((generic64_t *)f)[4ul] = 0ul;
    ((generic64_t *)f)[7ul] = 0ul;
    ((generic64_t *)f)[5ul] = 0ul;
    _var_25 = 0ul;
    if ((*(generic64_t *)&_var_44 & 4294967295ul) == 2ul) {
    } else {
      _var_25 = (generic64_t)len - ((generic64_t *)*(generic64_t *)&_var_43)[1ul];
    }
  }
  return (size_t)_var_25;
}

_ABI(SystemV_x86_64)
size_t unreserved___fwritex(typedef_312 s, size_t l, typedef_303 f) {
  struct _PACKED struct_581 {
    uint8_t padding_at_0[40];
  };
  generic64_t _var_0;
  generic64_t _var_1;
  generic64_t _var_2;
  generic64_t _var_3;
  generic64_t _var_4;
  generic8_t _var_5[4];
  generic8_t _var_6[8];
  generic8_t _var_7[8];
  generic8_t _var_8[8];
  generic8_t _var_9;
  generic8_t _var_10;
  if (((generic64_t *)f)[4ul] == 0ul) {
    *(int32_t *)&_var_5 = unreserved___towrite((FILE_ *)f);
    _var_0 = 0ul;
    if (*(generic32_t *)&_var_5 == 0u) {
      if ((generic8_t)((uint64_t)(((generic64_t *)f)[4ul] - ((generic64_t *)f)[5ul]) < (uint64_t)l)) {
        _var_0 = ((generic64_t *)f)[9ul];
      } else {
        _var_9 = (generic8_t)((int8_t)((generic8_t *)f)[139ul] < 0) ? 1u : (generic8_t)((generic64_t)l == 0ul);
        _var_1 = (generic64_t)s;
        _var_2 = (generic64_t)l;
        if (_var_9) {
          *(struct struct_718 **)&_var_8 = memcpy((struct struct_718 *)((generic64_t *)f)[5ul], (union union_596 *)_var_1, _var_2);
          ((generic64_t *)f)[5ul] = ((generic64_t *)f)[5ul] + _var_2;
          _var_0 = (generic64_t)l;
        } else {
          _var_3 = 0ul;
          _var_4 = (generic64_t)l;
        _label_0:
          if (*(generic8_t *)((generic64_t)l + (generic64_t)s + (_var_3 ^ 18446744073709551615ul)) == 10u) {
            struct rawfunction_25 _var_11 = ((rawfunction_25 *)((generic64_t *)f)[9ul])((pointer_or_number64_t)/* undef */ (generic64_t){0}, (pointer_or_number64_t)_var_4, (pointer_or_number64_t)s, (pointer_or_number64_t)f, (pointer_or_number64_t)/* undef */ (generic64_t){0}, (pointer_or_number64_t)/* undef */ (generic64_t){0});
            *(pointer_or_number64_t *)&_var_6 = _var_11.;
            *(pointer_or_number64_t *)&_var_7 = _var_11.;
            _var_0 = _var_4;
            if ((generic8_t)((uint64_t)_var_4 > (uint64_t)*(generic64_t *)&_var_6)) {
            } else {
              _var_2 = (generic64_t)l - _var_4;
              _var_1 = (generic64_t)l + (generic64_t)s - _var_3;
            }
          } else {
            _var_4 = _var_4 + 18446744073709551615ul;
            *(int8_t *)&_var_10 = (_var_3 ^ 18446744073709551615ul) == 0ul - (generic64_t)l;
            _var_3 = _var_3 + 1ul;
            if (_var_10) {
              _var_1 = (generic64_t)s;
              _var_2 = (generic64_t)l;
            } else {
              goto _label_0;
            }
          }
        }
      }
    } else {
    }
  } else {
  }
  return (size_t)_var_0;
}

_ABI(SystemV_x86_64)
size_t fwrite_unlocked(typedef_314 src, size_t size, size_t nmemb, typedef_303 f) {
  struct _PACKED struct_582 {
    uint8_t padding_at_0[16];
    union _PACKED union_725 {
      struct_724 member_0;
      struct_650 member_1;
    } *offset_16;
    uint8_t padding_at_24[32];
  };
  generic8_t _var_0[56];
  generic64_t _var_1;
  generic8_t _var_2;
  generic8_t _var_3[4];
  generic8_t _var_4[8];
  generic8_t _var_5;
  ((generic64_t *)&_var_0)[2ul] = (generic64_t)f;
  _var_5 = (generic8_t)((int32_t)((generic32_t *)f)[35ul] > 4294967295);
  _var_2 = 1u;
  if (_var_5) {
    *(int32_t *)&_var_3 = unreserved___lockfile((FILE_ *)f);
    *(int8_t *)&_var_2 = *(generic32_t *)&_var_3 == 0u;
  } else {
  }
  *(size_t *)&_var_4 = unreserved___fwritex((typedef_312)src, (size_t)((generic64_t)nmemb * (generic64_t)size), f);
  if (_var_2) {
  } else {
    unreserved___unlockfile((FILE_ *)f);
  }
  _var_1 = (generic64_t)nmemb;
  if ((generic64_t)nmemb * (generic64_t)size == *(generic64_t *)&_var_4) {
  } else {
    _var_1 = (generic64_t)((uint64_t)*(generic64_t *)&_var_4 / (uint64_t)size);
  }
  return (size_t)_var_1;
}

_ABI(SystemV_x86_64)
void unreserved___towrite_needs_stdio_exit(void) {
  struct _PACKED struct_583 {
    uint8_t padding_at_0[8];
  };
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
  generic8_t _var_25[8];
  generic8_t _var_26[4];
  generic8_t _var_27[8];
  generic8_t _var_28[8];
  generic8_t _var_29[8];
  generic8_t _var_30[8];
  *(FILE_ ***)&_var_25 = unreserved___ofl_lock();
  if (*(generic64_t *)*(generic64_t *)&_var_25 == 0ul) {
    close_file((FILE_ *)*(generic64_t *)4224952ul);
    if (*(generic64_t *)4223040ul == 0ul) {
    } else {
      if ((generic8_t)((int32_t)((generic32_t *)*(generic64_t *)4223040ul)[35ul] > 4294967295)) {
        *(int32_t *)&_var_26 = unreserved___lockfile((FILE_ *)*(generic64_t *)4223040ul);
      } else {
      }
      if ((generic8_t)((uint64_t)((generic64_t *)*(generic64_t *)4223040ul)[5ul] > (uint64_t)((generic64_t *)*(generic64_t *)4223040ul)[7ul])) {
        struct rawfunction_25 _var_31 = ((rawfunction_25 *)((generic64_t *)*(generic64_t *)4223040ul)[9ul])((pointer_or_number64_t)/* undef */ (generic64_t){0}, 0ul, 0ul, (pointer_or_number64_t)*(generic64_t *)4223040ul, (pointer_or_number64_t)/* undef */ (generic64_t){0}, (pointer_or_number64_t)/* undef */ (generic64_t){0});
        *(pointer_or_number64_t *)&_var_27 = _var_31.;
        *(pointer_or_number64_t *)&_var_28 = _var_31.;
      } else {
      }
      if ((generic8_t)((uint64_t)((generic64_t *)*(generic64_t *)4223040ul)[1ul] < (uint64_t)((generic64_t *)*(generic64_t *)4223040ul)[2ul])) {
        helper_syscall_wrapper((void *)0ul, 2u, 4209291ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 8ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)(int64_t)(int32_t)((generic32_t *)*(generic64_t *)4223040ul)[30ul], 1ul, ((generic64_t *)*(generic64_t *)4223040ul)[1ul] - ((generic64_t *)*(generic64_t *)4223040ul)[2ul], 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
        if ((generic8_t)((uint64_t)_var_3 > 18446744073709547520ul)) {
          *(void **)&_var_30 = (void *)(_var_4 + 18446744073709551600ul);
          *(generic64_t *)*(void **)&_var_30 = _var_3;
          *(int32_t **)&_var_29 = unreserved___errno_location();
          *(generic32_t *)*(generic64_t *)&_var_29 = 0u - (generic32_t)*(generic64_t *)*(void **)&_var_30;
        } else {
        }
      } else {
      }
    }
  } else {
    _var_24 = *(generic64_t *)*(generic64_t *)&_var_25;
  _label_0:
    close_file((FILE_ *)_var_24);
    _var_24 = ((generic64_t *)_var_24)[14ul];
    if (_var_24 == 0ul) {
    } else {
      goto _label_0;
    }
  }
  return;
}

_ABI(SystemV_x86_64)
void unreserved___lock(typedef_407 *l) {
  struct _PACKED struct_586 {
    generic64_t offset_0;
    uint8_t padding_at_8[8];
    generic64_t offset_16;
  };
  generic8_t _var_0[24];
  generic8_t _var_1[4];
  generic8_t _var_2[4];
  if (*(generic32_t *)4225036ul == 0u) {
  } else {
    ((generic64_t *)&_var_0)[2ul] = 1ul;
    *(generic64_t *)&_var_0 = 1ul;
    helper_lock();
    *(generic32_t *)&_var_1 = *(generic32_t *)l;
    *(generic32_t *)l = 1u;
    helper_unlock();
    if (*(generic32_t *)&_var_1 == 0u) {
    } else {
    _label_0:
      unreserved___wait((typedef_315 *)l, (typedef_315 *)l + 1ul, 1, 1);
      helper_lock();
      *(generic32_t *)&_var_2 = *(generic32_t *)l;
      *(generic32_t *)l = 1u;
      helper_unlock();
      if (*(generic32_t *)&_var_2 == 0u) {
      } else {
        goto _label_0;
      }
    }
  }
  return;
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
  if (*(generic32_t *)4224960ul == 0u) {
  } else {
    *(generic32_t *)4224960ul = 0u;
    helper_lock();
    helper_unlock();
    if (*(generic32_t *)4224964ul == 0u) {
    } else {
      helper_syscall_wrapper((void *)0ul, 2u, 4210742ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 4224960ul, 1ul, 129ul, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42, (void *)&_var_43, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47);
      if (_var_27 == 18446744073709551578ul) {
        helper_syscall_wrapper((void *)0ul, 2u, 4210756ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, 4224960ul, 1ul, 1ul, _var_29, _var_30, _var_32, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, _var_34, _var_38, _var_41, _var_42, 274877906944ul, 127u, 2147549185ul, 0ul, _var_43, _var_45, _var_46, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
      } else {
      }
    }
  }
  return;
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
  if (*(generic32_t *)l == 0u) {
  } else {
    *(generic32_t *)l = 0u;
    helper_lock();
    helper_unlock();
    if (((generic32_t *)l)[1ul] == 0u) {
    } else {
      helper_syscall_wrapper((void *)0ul, 2u, 4210742ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)l, 1ul, 129ul, 4294967295u, 514ul, 4243635u, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, 0ul, 0ul, 0ul, 65535u, 274877906944ul, 127u, 2147549185ul, 0ul, 0u, 0ul, 4294967295u, (void *)&_var_24, (void *)&_var_25, (void *)&_var_26, (void *)&_var_27, (void *)&_var_28, (void *)&_var_29, (void *)&_var_30, (void *)&_var_31, (void *)&_var_32, (void *)&_var_33, (void *)&_var_34, (void *)&_var_35, (void *)&_var_36, (void *)&_var_37, (void *)&_var_38, (void *)&_var_39, (void *)&_var_40, (void *)&_var_41, (void *)&_var_42, (void *)&_var_43, (void *)&_var_44, (void *)&_var_45, (void *)&_var_46, (void *)&_var_47);
      if (_var_27 == 18446744073709551578ul) {
        helper_syscall_wrapper((void *)0ul, 2u, 4210756ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, 202ul, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, /* undef */ (generic64_t){0}, (generic64_t)l, 1ul, 1ul, _var_29, _var_30, _var_32, 0ul, 0ul, 15727360u, 0ul, 13628160u, 0ul, _var_34, _var_38, _var_41, _var_42, 274877906944ul, 127u, 2147549185ul, 0ul, _var_43, _var_45, _var_46, (void *)&_var_0, (void *)&_var_1, (void *)&_var_2, (void *)&_var_3, (void *)&_var_4, (void *)&_var_5, (void *)&_var_6, (void *)&_var_7, (void *)&_var_8, (void *)&_var_9, (void *)&_var_10, (void *)&_var_11, (void *)&_var_12, (void *)&_var_13, (void *)&_var_14, (void *)&_var_15, (void *)&_var_16, (void *)&_var_17, (void *)&_var_18, (void *)&_var_19, (void *)&_var_20, (void *)&_var_21, (void *)&_var_22, (void *)&_var_23);
      } else {
      }
    }
  }
  return;
}

