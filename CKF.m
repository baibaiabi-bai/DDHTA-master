function [Pyy2,v,P_k,X_k,HPH]=CKF(Xjian,Pk_1,zk,Q,R,T,nn)
    dt=T;
    n=size(Xjian,1);
    m = 2*n;  
    kesai=sqrt(m/2) * [eye(n),-eye(n)];
    w=1/m;
    
    [U,S,V]=svd(Pk_1);
    S1=U*S^0.5*V';
    
    Xr=S1*kesai+repmat(Xjian,1,m);
    
    
    for i=1:m
        Xcr(:,i) =f(Xr(:,i),nn,dt) ;  
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
    
    HPH = (1/m)*(yr*yr') - Ypre*Ypre'; 
    Pyy2 = HPH + R; 
   
    Pyy=(1/m)*(yr*yr')-Ypre*Ypre'+R;
    
    Pxy=(1/m)*Xp*yr'-Xpr*Ypre';
    K = Pxy*inv(Pyy); 
    v=zk-Ypre;
    X_k=Xpr+K*(zk-Ypre);                 
    P_k=Ppr-K*Pyy*K';  
end