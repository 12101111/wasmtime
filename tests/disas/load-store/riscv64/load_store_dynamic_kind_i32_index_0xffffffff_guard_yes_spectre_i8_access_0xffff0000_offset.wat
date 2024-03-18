;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

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
;;   ld a4,88(a0)
;;   ld a5,80(a0)
;;   slli a0,a2,32
;;   srli a0,a0,32
;;   sltu a4,a4,a0
;;   add a5,a5,a0
;;   lui a0,65535
;;   slli a0,a0,4
;;   add a5,a5,a0
;;   sub a2,zero,a4
;;   not a4,a2
;;   and a0,a5,a4
;;   sb a3,0(a0)
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
;;   ld a4,88(a0)
;;   ld a5,80(a0)
;;   slli a3,a2,32
;;   srli a0,a3,32
;;   sltu a4,a4,a0
;;   add a5,a5,a0
;;   lui a3,65535
;;   slli a0,a3,4
;;   add a5,a5,a0
;;   sub a2,zero,a4
;;   not a4,a2
;;   and a0,a5,a4
;;   lbu a0,0(a0)
;;   j label1
;; block1:
;;   ld ra,8(sp)
;;   ld fp,0(sp)
;;   addi sp,sp,16
;;   ret
