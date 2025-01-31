declare !revng.tags !0 void @scope-closer(ptr)
declare !revng.tags !1 void @goto-block()
!0 = !{!"scope-closer"}
!1 = !{!"goto-block"}

define void @f(i1 noundef %a) !revng.function.entry !{!"0x40001001:Code_x86_64"} {
block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br label %block_d

block_c:
  br label %block_d

block_d:
  ret void
}
