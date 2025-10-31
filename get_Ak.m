function Ak_seq=get_Ak(Mkc_seq,xk,Pk,Mk_1_seq,mu,Q,R,zk,T)

n1=length(Mk_1_seq);
Mk_pre_seq=Mk_1_seq;
n2=length(Mkc_seq);

for ii=1:n1
    [Pk_sk(:,:,ii),zk_j(:,ii)]=mean_pre(xk,Pk,zk,Q(:,:,Mk_pre_seq(ii)),R,Mk_pre_seq(ii),T);
% zk_j(:,ii)
end
yk_sk=mu*zk_j';

for jj=1:n1
  PP(:,:,jj)=mu(1,jj)*(Pk_sk(:,:,jj)+(zk_j(:,ii)-yk_sk')*(zk_j(:,ii)-yk_sk')');
  
end
Pk_sk=sum(PP,3);
%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:n2
    [Pk_j(:,:,ii),yk_j(:,ii)]=mean_pre(xk,Pk,zk,Q(:,:,Mkc_seq(ii)),R,Mkc_seq(ii),T);

end

lamda=0.5;

for jj=1:n2
Renyi(1,jj)=-1/2/(1-lamda)*log((det(Pk_j(:,:,jj)))^lamda*(det(Pk_sk))^(1-lamda)/det(lamda*Pk_j(:,:,jj)+(1-lamda)*Pk_sk))+...
    lamda/2*(yk_sk'-yk_j(:,jj))'*inv(lamda*Pk_j(:,:,jj)+(1-lamda)*Pk_sk)*(yk_sk'-yk_j(:,jj));
end
nq=3;
[~, min_indices] = mink(Renyi,nq);
%Ak_seq=[Mkc_seq(min_indices(1)) Mkc_seq(min_indices(2)) Mkc_seq(min_indices(3))];
Ak_seq=[Mkc_seq(min_indices(1))];
end