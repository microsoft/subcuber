from dsl import *

def strassen_winograd(m,n,k):
    a = Matrix((m,k))
    b = Matrix((k,n))

    a0 = a[:m//2,:k//2]
    a1 = a[:m//2,k//2:]
    a2 = a[m//2:,:k//2]
    a3 = a[m//2:,k//2:]
    b0 = b[:k//2,:n//2]
    b1 = b[:k//2,n//2:]
    b2 = b[k//2:,:n//2]
    b3 = b[k//2:,n//2:]

    s1 = a2+a3
    s2 = s1-a0
    b10 = b1-b0
    b31 = b3-b1
    s3 = b31+b0

    a02 = a0-a2
    a1s2 = a1-s2

    s3b2 = s3-b2

    m0 = a0@b0       #Mem Access: (M/2/TM * N/2/TN) * K/2 * (TM + TN)            ; 512
    m1 = a1@b2       #Mem Access: (M/2/TM * N/2/TN) * K/2 * (TM + TN)            ; 512
    m2 = (s2)@(s3)   #Mem Access: (M/2/TM * N/2/TN) * K/2 * (3*TM + 3*TN) = 3*M0 ; (1+1-1=1)@(1-(1-1)=1)= 512
    m3 = (a02)@(b31) #Mem Access: (M/2/TM * N/2/TN) * K/2 * (2*TM + 2*TN) = 2*M0 ; (1-1)@(1-1) 0
    m4 = (s1)@(b10)  #Mem Access: (M/2/TM * N/2/TN) * K/2 * (2*TM + 2*TN) = 2*M0 ; (1+1)@(1-1) 0
    m5 = (a1s2)@b3   #Mem Access: (M/2/TM * N/2/TN) * K/2 * (4*TM + TN)          ; (1-(1+1-1)=0)@1 = 0@1 = 0
    m6 = a3@(s3b2)   #Mem Access: (M/2/TM * N/2/TN) * K/2 * (TM + 4*TN)          ; 1@(1-()-1) = 0

    v1 = m0+m2
    v2 = v1+m3
    v3 = v1+m4

    c0 = m0+m1
    c1 = v3+m5 #m0+m2+m4+m5 ; 512+512+0+0+0 = 1024
    c2 = v2-m6 #m0+m2+m3-m6 ; 
    c3 = v2+m4 #m0+m2+m3+m4 ;

    c = Combine4x4([c0, c1, c2, c3])
    c.var_names(locals())
    # c.print_tree()

    return c,[m0,m1,m2,m3,m4,m5,m6]