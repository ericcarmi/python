# IFS structures/matrices
from numpy import array,pi,sin,cos,sqrt

def getIFS(name):
    if(name == 'Sierpinski'):
        a=[0.5,0.5,0.5]
        b=[0,0,0]
        c=[0,0,0]
        d=[0.5,0.5,0.5]
        E=[0,0,0.5]
        F=[0.5,0,0]
        A1 = array(([a[0], b[0]], [c[0], d[0]]))
        A2 = array(([a[1], b[1]], [c[1], d[1]]))
        A3 = array(([a[2], b[2]], [c[2], d[2]]))
        T1 = array(([E[0]],[F[0]]))
        T2 = array(([E[1]],[F[1]]))
        T3 = array(([E[2]],[F[2]]))
        p=[1/3.,1/3.,1/3.]
        A=[A2,A3,A1] # Swapped to make index correspond to address 
        T=[T1,T2,T3]
        return A,T,p

    elif(name == "tree"):
        a=[0,0.42,0.42,0.1]
        b=[0,-0.42,0.42,0]
        c=[0,0.42,-0.42,0]
        d=[0.5,0.42,0.42,0.1]
        E=[0,0,0,0]
        F=[0,0.2,0.2,0.2]
        p=[0.05,0.4,0.4,0.15]

        A1 = array(([a[0], b[0]], [c[0], d[0] ]))
        A2 = array(([a[1], b[1]], [c[1], d[1] ]))
        A3 = array(([a[2], b[2]], [c[2], d[2] ]))
        A4 = array(([a[3], b[3]], [c[3], d[3] ]))
        T1 = array(([E[0]],[F[0]]))
        T2 = array(([E[1]],[F[1]]))
        T3 = array(([E[2]],[F[2]]))
        T4 = array(([E[3]],[F[3]]))
        A=[A1,A2,A3,A4]
        T=[T1,T2,T3,T4]
        return A,T,p

    elif(name == "fern"):
        a=[0,0.85,0.2,-0.15]
        b=[0,0.04,-0.26,0.28]
        c=[0,-0.04,0.23,0.26]
        d=[0.16,0.85,0.22,0.24]
        E=[0,0,0,0]
        F=[0,1.6,1.6,0.44]
        p=[0.01,0.85,0.07,0.07]

        A1 = array(([a[0], b[0]], [c[0], d[0] ]))
        A2 = array(([a[1], b[1]], [c[1], d[1] ]))
        A3 = array(([a[2], b[2]], [c[2], d[2] ]))
        A4 = array(([a[3], b[3]], [c[3], d[3] ]))
        T1 = array(([E[0]],[F[0]]))
        T2 = array(([E[1]],[F[1]]))
        T3 = array(([E[2]],[F[2]]))
        T4 = array(([E[3]],[F[3]]))
        A=[A1,A2,A3,A4]
        T=[T1,T2,T3,T4]
        return A,T,p

    elif(name == "dragon"):
        x = 0.5
        A1 = array(([x, -x],  [x, x]))
        A2 = array(([-x, -x], [x, -x]))
        T1 = array(([0] ,[0]))
        T2 = array(([1] ,[0]))
        A=[A1,A2]
        T=[T1,T2]
        p=[0.5,0.5]
        return A,T,p
