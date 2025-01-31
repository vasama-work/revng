declare !revng.tags !0 void @scope-closer(ptr)
declare !revng.tags !1 void @goto-block()
!0 = !{!"scope-closer"}
!1 = !{!"goto-block"}

define void @f(i1 noundef %a, i1 noundef %b) !revng.function.entry !{!"0x40001002:Code_x86_64"} {

block_a:
  br i1 %a, label %block_b, label %block_d

block_b:
  br i1 %b, label %block_c, label %goto_d

block_c:
  br label %block_e

goto_d:
  call void @goto-block()
  call void @scope-closer(ptr blockaddress(@f, %block_e))
  br label %block_d

block_d:
  br label %block_e

block_e:
  ret void
}
