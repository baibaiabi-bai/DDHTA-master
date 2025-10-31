function [Pyy,Ypre]=mean_pre(Xjian,Pk_1,zk,Q,R,flag,T)

    dt=T;
    
    n=size(Xjian,1);
    m = 2*n;  
    kesai=sqrt(m/2) * [eye(n),-eye(n)];
    w=1/m;
    
    [U,S,V]=svd(Pk_1);
    S1=U*S^0.5*V';
    Xr=S1*kesai+repmat(Xjian,1,m);
    
    Xcr=zeros(n,m);
    for i=1:m
        Xcr(:,i) =f(Xr(:,i),flag,T) ; 
    end
    
    Xpr=(1/m).*sum(Xcr,2);
    
    Ppr=(1/m).*(Xcr*Xcr')-Xpr*Xpr'+ Q;
    
    Pxx=(1/m).*(Xr*Xcr')-Xjian*Xpr';
    [U,S,V]=svd(Ppr);
    S2=U*S^0.5*V';
    Xp=S2*kesai+repmat(Xpr,1,m);
    var1=size(zk,1);
    yr=zeros(var1,m);
    for i=1:m
        yr(:,i) = h_vsimm(Xp(:,i));  
    end
    
    Ypre = (1/m).*sum(yr,2);
    Pyy=(1/m)*(yr*yr')-Ypre*Ypre'+R;

end