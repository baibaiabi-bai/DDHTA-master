function  [mu_j,v,S,x_k,Pk_j,xk,Pk,HPH,v_n]=VSIMM(i,Mk_seq,Mk_1_seq,xk_1_i,mu,Pk_1_i,zk,...
    Q,R,model_n,T)

    P=@(a,n)diag(a*ones(1,n))+(1-a)/(n-1)*ones(n,n)-(1-a)/(n-1)*diag(ones(1,n));
    P=P(0.9,model_n);
    n1=length(Mk_1_seq);
    n2=length(Mk_seq);
    length_window=8;
    
    for jj=1:n2
        for ii=1:n1
            u_m(ii,jj)=P(Mk_1_seq(ii),Mk_seq(jj))*mu(ii);
        end
      u(1,jj)=sum(u_m(:,jj));
    end
    
    %%%%%%%Interactive weight%%%%%
    for jj=1:n2
        for ii=1:n1
            u_m_ij(ii,jj)=P(Mk_1_seq(ii),Mk_seq(jj))*mu(ii)/u(jj);%%
        end
    end
    
    
    %%%%%%%Interaction estimation%%%%%%%%%%
    for jj=1:n2
        for ii=1:n1
            xk_1_j(:,ii,jj)= u_m_ij(ii,jj)*xk_1_i(:,ii);
        end 
        xk_pj(:,jj)=sum(xk_1_j(:,:,jj),2);
    end
    
    %%%%%%%%Interaction variance%%%%%%%%%
    
    for jj=1:n2
        for ii=1:n1
    
            Pk_1_j(:,:,ii,jj)= u_m_ij(ii,jj)*(Pk_1_i(:,:,ii)+(xk_1_i(:,ii)-xk_pj(:,jj))*(xk_1_i(:,ii)-xk_pj(:,jj))');
      
        end
       Pk_pj(:,:,jj)=sum(Pk_1_j(:,:,:,jj),3);
    end
    
    %%%%%%%%%%%%%%%%%%%%%
    for jj=1:n2
        
        [S(:,:,jj),v(:,jj),Pk_j(:,:,jj),x_k(:,jj)]=CKF(xk_pj(:,jj),Pk_pj(:,:,jj),zk,Q(:,:,jj),R,T,Mk_seq(jj));
        
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%Solving likelihood function%%%%%%%%
    
    
    for jj=1:n2
       like(jj,1)=1/sqrt((2*pi*det(S(:,:,jj))))*exp(-0.5*v(:,jj)'*inv(S(:,:,jj))*v(:,jj));%似然
    end
     
    
    var3=like'*u';
    
    for jj=1:n2
          mu_j(1,jj)=like(jj,1)*u(1,jj)/var3;
    end
    
    %%%%%%%%State fusion%%%%%%%%%%%%%%%%
    xk=mu_j(1,:)*x_k(:,:)';
         
    HPH = zeros(3);
    for j = 1:n2
        HPH = HPH + mu(j) * S(:,:,j);
    end
    v_n = zeros(1);
    for j = 1:n2
        v_n = v_n + mu(j) * v(:,j);
    end
    
    %%%%%%%%Overall covariance estimation%%%%%%%%%%%%%%%%%
    for jj=1:n2
      PP(:,:,jj)=mu_j(1,jj)*(Pk_j(:,:,jj)+(xk-x_k(:,jj))*(xk-x_k(:,jj))');
    end
    Pk=sum(PP,3);
end