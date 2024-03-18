;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

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
;;   stp fp, lr, [sp, #-16]!
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mov fp, sp
;;   ldr x16, [x0, #8]
;;   ldr x16, [x16]
;;   subs xzr, sp, x16, UXTX
;;   b.lo #trap=stk_ovf
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   mov w11, w2
;;   movn w12, #65534
;;   adds x11, x11, x12
;;   b.hs #trap=heap_oob
;;   ldr x12, [x0, #88]
;;   subs xzr, x11, x12
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x14, [x0, #80]
;;   add x14, x14, x2, UXTW
;;   movz x15, #65535, LSL #16
;;   strb w3, [x14, x15]
;;   b label2
;; block2:
;;   ldp fp, lr, [sp], #16
;;   ret
;; block3:
;;   udf #0xc11f
;;
;; function u0:1:
;;   stp fp, lr, [sp, #-16]!
;;   unwind PushFrameRegs { offset_upward_to_caller_sp: 16 }
;;   mov fp, sp
;;   ldr x16, [x0, #8]
;;   ldr x16, [x16]
;;   subs xzr, sp, x16, UXTX
;;   b.lo #trap=stk_ovf
;;   unwind DefineNewFrame { offset_upward_to_caller_sp: 16, offset_downward_to_clobbers: 0 }
;; block0:
;;   mov w11, w2
;;   movn w12, #65534
;;   adds x11, x11, x12
;;   b.hs #trap=heap_oob
;;   ldr x12, [x0, #88]
;;   subs xzr, x11, x12
;;   b.hi label3 ; b label1
;; block1:
;;   ldr x14, [x0, #80]
;;   add x14, x14, x2, UXTW
;;   movz x15, #65535, LSL #16
;;   ldrb w0, [x14, x15]
;;   b label2
;; block2:
;;   ldp fp, lr, [sp], #16
;;   ret
;; block3:
;;   udf #0xc11f
