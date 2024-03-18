;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; function u0:0:
;;   addi sp,sp,-16
;;   sd ra,8(sp)
;;   sd fp,0(sp)
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mv fp,sp
;;   ld t6,8(a0)
;;   ld t6,0(t6)
;;   trap_if stk_ovf##(sp ult t6)
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   ld a1,88(a0)
;;   ld a0,80(a0)
;;   sltu a5,a2,a1
;;   xori a1,a5,1
;;   add a0,a0,a2
;;   sub a4,zero,a1
;;   not a1,a4
;;   and a2,a0,a1
;;   sb a3,0(a2)
;;   j label1
;; block1:
;;   ld ra,8(sp)
;;   ld fp,0(sp)
;;   addi sp,sp,16
;;   ret
;;
;; function u0:1:
;;   addi sp,sp,-16
;;   sd ra,8(sp)
;;   sd fp,0(sp)
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mv fp,sp
;;   ld t6,8(a0)
;;   ld t6,0(t6)
;;   trap_if stk_ovf##(sp ult t6)
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   ld a1,88(a0)
;;   ld a0,80(a0)
;;   sltu a5,a2,a1
;;   xori a1,a5,1
;;   add a0,a0,a2
;;   sub a4,zero,a1
;;   not a1,a4
;;   and a2,a0,a1
;;   lbu a0,0(a2)
;;   j label1
;; block1:
;;   ld ra,8(sp)
;;   ld fp,0(sp)
;;   addi sp,sp,16
;;   ret
