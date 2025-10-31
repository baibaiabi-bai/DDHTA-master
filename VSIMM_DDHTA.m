clear
close all

load('test_3.mat');
load('residual_norm_params.mat'); 
load('normalization_params.mat'); %%%%%%data loading

loaded_model = load('DDHTA_net.mat');
track_num = 3;
net = loaded_model.net;

length_window=40;%parameter initialization
MC = 100;
range_max = 6.5e5;
dt = 0.1;
model_n=9;
len = size(X,2);
t = dt:dt:len*dt;

az = measurement(1,:);
elev = measurement(2,:);
slantRange = measurement(3,:);

DCJA1=1;   
DSTNet1=1;   
sava_data = 0;  

Q = cell(model_n, 1);
G = cell(model_n, 1);
DDHTA = cell(MC,1);

x_left = 500;
x_m = 1e3;
x_right = 2000 ;
x_a = [x_left,x_right];
y_left =300;
y_m =1e3;
y_right = 2000 ;
y_a = [y_left,y_right];
z_left = 300 ;
z_m = 1e3 ;
z_right =2000 ;

z_a = [z_left,z_right];
q_all = allcomb(x_a, y_a, z_a);
q_all(9,:) = [1,1,1];

for g=1:9
    Q{g,1} = diag(q_all(g,:));
end


var1 = 0.9;
var2 = 10;

az_std = rad2deg(1e-3);
elev_std = rad2deg(1e-3);
rho_std = 50;

beta = 0.95;      
min_samples = 50; 
R_adapt_rate = 0.2;     

v_history = [];        
S_hat_history = [];    
g = 0;
n_z = 3; 


gamma_base = 0.95;    
alpha = 0.3;         
T_D = chi2inv(1-alpha, n_z); 
R_hat = diag([az_std^2, elev_std^2, rho_std^2]);

R_prior = diag(R_hat);
R_prior1 = R_hat;
r_prior_diag = diag(R_prior1); 
R_history = repmat(r_prior_diag, 1, len);

rng('shuffle');
for mc = 1:MC
    mc
    
    %add noise
    p_abnormal = 1 - 0.9; 
    normal_std_multiplier = 1;
    abnormal_std_multiplier = sqrt(10);

    is_abnormal_az = (rand(1, size(az, 2)) < p_abnormal);
    noise_az = normal_std_multiplier * az_std * randn(1, size(az, 2));
    noise_az(is_abnormal_az) = abnormal_std_multiplier * az_std * randn(1, sum(is_abnormal_az));
    az_n = az + noise_az;

    is_abnormal_elev = (rand(1, size(elev, 2)) < p_abnormal);
    noise_elev = normal_std_multiplier * elev_std * randn(1, size(elev, 2));
    noise_elev(is_abnormal_elev) = abnormal_std_multiplier * elev_std * randn(1, sum(is_abnormal_elev));
    elev_n = elev + noise_elev;


    is_abnormal_sr = (rand(1, size(slantRange, 2)) < p_abnormal);
    noise_sr = normal_std_multiplier * rho_std * randn(1, size(slantRange, 2));
    noise_sr(is_abnormal_sr) = abnormal_std_multiplier * rho_std * randn(1, sum(is_abnormal_sr));
    slantRange_n = slantRange + noise_sr;



    [east, north, up] = aer2enu(az_n, elev_n, slantRange_n);
    meas_ENU(1,:) = east;
    meas_ENU(2,1) = X(2,1)+5;
    meas_ENU(2,2:1500) = diff(east)/0.1;
    meas_ENU(3,:) = north;
    meas_ENU(4,1) = X(5,1)+5;
    meas_ENU(4,2:1500) = diff(north)/0.1;
    meas_ENU(5,:) = up;
    meas_ENU(6,1) = X(8,1)-2;
    meas_ENU(6,2:1500) = diff(up)/0.1;

    meas_nosies{mc,1}= meas_ENU;%Calculate measurement noise
    

    meas = [az_n; elev_n; slantRange_n];
    X0 = [X(1,1)+50, X(2,1)+5, X(3,1), X(4,1)-20, X(5,1)+5, X(6,1), X(7,1)+50, X(8,1)-2, X(9,1)];
    DDHTA{mc, 1}(:,1) = X0;
    P0 = diag([1e+4, 1e+3, 10, 1e+4, 1e+3, 10, 1e+4, 1e+3, 10]);
    for n = 1:model_n
        Pk_1_i(:, :, n) = P0;
    end
    xk_1_i = repmat(X0', 1, model_n);

    model_seq = 1:model_n;
   
    mu0=[1/3 1/3 1/3];
    mu_k_all_select = zeros(model_n,size(X,2));  

    Mk_seq = [1 2 3];
    Mk_1_seq = [1 2 4];

    mu_k_all_select(Mk_seq,1) = mu0;

    for m = 1:9
            if m==9
                G= get_G_matrix(dt,'CA');
                Q_all(:,:,m)= G(:,:)*Q{m,1}(:,:)* G(:,:)';
            else
                G= get_G_matrix(dt,'CV');
                Q_all(:,:,m)= G(:,:)*Q{m,1}(:,:)* G(:,:)';

            end
    end

    detection_flags = false(1, len); 
    for i = 2:len
        Mk_num = size(Mk_seq, 2);
        for m = 1:Mk_num
            if Mk_seq(m)==9
                G= get_G_matrix(dt,'CA');
                Q_update(:,:,m)= G(:,:)*Q{Mk_seq(m),1}(:,:)* G(:,:)';
            else
                G= get_G_matrix(dt,'CV');
                Q_update(:,:,m)= G(:,:)*Q{Mk_seq(m),1}(:,:)* G(:,:)';
            end
        end
        [mu_k, ~,~,x_k,Pk_j,xk,Pk,HPH,v_n(:,i)] = VSIMM(i, Mk_seq, Mk_1_seq, xk_1_i, mu0, Pk_1_i, meas(:,i),Q_update,R_prior1,model_n,dt);%Variable Structure Interactive Multi Model Filtering
        
        %Start DCJA
        if DCJA1== 1
            S_k_theoretical = HPH + R_prior1;  
           var = 0.95;
           if i==2
               S_hat_current=v_n(:,i) * v_n(:,i)';
           else
               S_hat_current=(var*S_hat_current_1+v_n(:,i) * v_n(:,i)')/(1+var); 
           end
           S_hat_current_1 = S_hat_current;
    
    
           
           current_diff = S_hat_current - S_k_theoretical;
           norm_diff = norm(current_diff, 'fro');
           norm_theoretical = norm(S_k_theoretical, 'fro');
           cond1 = norm_diff> alpha * norm_theoretical;
    
    
           
           cond2 = false;
           if ~isempty(S_hat_history)
                
                L = size(S_hat_history,3);
                weights = gamma_base.^(L-1:-1:0); 
                weights = weights / sum(weights); 
    
                S_hist = zeros(size(S_hat_history(:,:,1)));
                for k = 1:L
                    S_hist = S_hist + weights(k) * S_hat_history(:,:,k);
                end
    
              chi2_thresh = chi2inv(beta, size(v_n(:,i),1));
              mahalanobis_dist = v_n(:,i)' * inv(S_hist) * v_n(:,i);
              gamma = gamma_base * exp(-mahalanobis_dist / chi2_thresh);
              gamma = max(min(gamma, 0.99), 0.8);
    
              cond2 = mahalanobis_dist > chi2_thresh;
           end
    
           if i > 50 && cond1 && cond2  
          
               R_delta_full = R_adapt_rate * (S_hat_current - S_k_theoretical);
               R_hat = R_prior1 + R_delta_full;
               R_hat = [R_hat(1,1) 0 0; 0 R_hat(2,2) 0;0 0 R_hat(3,3)];
    
               r_prior_diag = diag(R_prior1); 

               R_hat = diag(min(max(diag(R_hat), r_prior_diag * 0.5), r_prior_diag * 10));
               g = g+1;  
                 [mu_k, ~,~,x_k,Pk_j,xk,Pk,HPH,v_n(:,i)] = VSIMM(i, Mk_seq, Mk_1_seq, xk_1_i, mu0, Pk_1_i, meas(:,i),Q_update,R_hat,model_n,dt);
          end
    
          if isempty(v_history) 
             v_history = v_n(:,i);
             S_hat_history = S_hat_current;
          else
             v_history = [v_history, v_n(:,i)]; 
             S_hat_history = cat(3, S_hat_history, S_hat_current);
    
             if size(v_history,2) > 100 
                 v_history = v_history(:,2:end);
                 S_hat_history = S_hat_history(:,:,2:end);
             end
          end
        end
       
        Pk_1_i = Pk_j;
        xk_1_i = x_k;
        mu0 = mu_k;
        Mk_1_seq = Mk_seq;
        mu_k_all_select(Mk_seq,i) = mu_k;
        mu_k_all(:,i) = mu_k;
        
        %Start DSTNet
        if i > 50 && DSTNet1== 1
           window_start = max(i-length_window, 1);
           residual_window = v_n(:, window_start:i); 
           normalized_window = (residual_window - residual_mean) ./ residual_std;
           predicted_delta = predict(net, {normalized_window});
           predicted_delta = predicted_delta .* error_std' + error_mean';
           %threshold = [10, 5, 2, 10, 5, 2, 10, 5, 2];
           %threshold = [8, 2, 0.5, 8, 2, 1, 8, 2, 0.5]; %Single step constraint
           %predicted_delta = min(max(predicted_delta, -threshold), threshold);

           xk = xk + predicted_delta;
          
        end
        DDHTA{mc, 1}(:,i) = xk';
     
        %Model set adaptation
        Mpk_seq=get_Mpk(Mk_1_seq,mu_k,1);
        Mkc_seq=get_Mkc_seq(model_seq,Mk_1_seq);
        if DCJA1== 1
            if i > 50 && cond1 && cond2 
                Ak_seq=get_Ak(Mkc_seq,xk',Pk,Mk_1_seq,mu_k,Q_all,R_hat,meas(:,i),dt);
            else
            Ak_seq=get_Ak(Mkc_seq,xk',Pk,Mk_1_seq,mu_k,Q_all,R_prior1,meas(:,i),dt);
            end
        else
            Ak_seq=get_Ak(Mkc_seq,xk',Pk,Mk_1_seq,mu_k,Q_all,R_prior1,meas(:,i),dt);
        end
        merged_seq = unique([Mpk_seq, Ak_seq]);
        Mk_seq=merged_seq;


     end

end

figure
plot3(X(1,:)/1e3,X(4,:)/1e3,X(7,:)/1e3,'g',LineWidth=1);
hold on
plot3(DDHTA{MC,1}(1,:)/1e3, DDHTA{MC,1}(4,:)/1e3, DDHTA{MC,1}(7,:)/1e3,'r',LineWidth=1);
hold on
xlabel('East/km');
ylabel('North/km');
zlabel('Up/km');
grid on
hold on

for ii= 1:len
        for iii=1:MC
           RMSE1_a(ii,iii)=(DDHTA{iii,1}(1,ii)-X(1,ii)).^2;
           RMSE2_a(ii,iii)=(DDHTA{iii,1}(4,ii)-X(4,ii)).^2; 
           RMSE3_a(ii,iii)=(DDHTA{iii,1}(7,ii)-X(7,ii)).^2;

           RMSE123_a(ii,iii)=RMSE1_a(ii,iii) + RMSE2_a(ii,iii) + RMSE3_a(ii,iii);

           RMSE4_b(ii,iii)=(DDHTA{iii,1}(2,ii)-X(2,ii)).^2; 
           RMSE5_b(ii,iii)=(DDHTA{iii,1}(5,ii)-X(5,ii)).^2; 
           RMSE6_b(ii,iii)=(DDHTA{iii,1}(8,ii)-X(8,ii)).^2;
           RMSE456_a(ii,iii)=RMSE4_b(ii,iii) + RMSE5_b(ii,iii) + RMSE6_b(ii,iii);
        end
        RMSEa=sqrt(sum(RMSE1_a,2)/MC);
        RMSEb=sqrt(sum(RMSE2_a,2)/MC);
        RMSEc=sqrt(sum(RMSE3_a,2)/MC);

        DDHTA_pos = sqrt(sum(RMSE123_a,2)/MC);
        DDHTA_v = sqrt(sum(RMSE456_a,2)/MC);
        RMSE4b=sqrt(sum(RMSE4_b,2)/MC);
        RMSE5b=sqrt(sum(RMSE5_b,2)/MC);
        RMSE6b=sqrt(sum(RMSE6_b,2)/MC);
end


for ii= 1:len
        for iii=1:MC
           RMSE1_a(ii,iii)=(meas_nosies{iii,1}(1,ii)-X(1,ii)).^2;
           RMSE2_a(ii,iii)=(meas_nosies{iii,1}(3,ii)-X(4,ii)).^2; 
           RMSE3_a(ii,iii)=(meas_nosies{iii,1}(5,ii)-X(7,ii)).^2;

           RMSE123_a(ii,iii)=RMSE1_a(ii,iii) + RMSE2_a(ii,iii) + RMSE3_a(ii,iii);

           RMSE4_b(ii,iii)=(meas_nosies{iii,1}(2,ii)-X(2,ii)).^2; 
           RMSE5_b(ii,iii)=(meas_nosies{iii,1}(4,ii)-X(5,ii)).^2; 
           RMSE6_b(ii,iii)=(meas_nosies{iii,1}(6,ii)-X(8,ii)).^2;
           RMSE456_a(ii,iii)=RMSE4_b(ii,iii) + RMSE5_b(ii,iii) + RMSE6_b(ii,iii);
        end
        RMSEa=sqrt(sum(RMSE1_a,2)/MC);
        RMSEb=sqrt(sum(RMSE2_a,2)/MC);
        RMSEc=sqrt(sum(RMSE3_a,2)/MC);

        meas_nosies_pos = sqrt(sum(RMSE123_a,2)/MC);
        meas_nosies_v = sqrt(sum(RMSE456_a,2)/MC);
        RMSE4b=sqrt(sum(RMSE4_b,2)/MC);
        RMSE5b=sqrt(sum(RMSE5_b,2)/MC);
        RMSE6b=sqrt(sum(RMSE6_b,2)/MC);
end


figure
plot(t,DDHTA_pos)
hold on 
% plot(t,meas_nosies_pos)
xlabel('t/s')
ylabel('errors')
title("Position RMSE")


mean_pos = mean(DDHTA_pos,1)
mean_v = mean(DDHTA_v,1)
std_pos = std(DDHTA_pos,1)
std_v = std(DDHTA_v,1)
RMSE_pos = 1/len*(sum(DDHTA_pos,1))
RMSE_v = 1/len*(sum(DDHTA_v,1))


%Save errors
 if sava_data==1 && MC==100
    if DCJA1 ==1 && DSTNet1~=1
        DCJA=DDHTA;
        DCJA_pos=DDHTA_pos;
        DCJA_v=DDHTA_v;
        output_dir = 'D:\Desktop\other1.3\paper1\testCode\CKF_IMM_VSIMM\figure';
        filename = sprintf('DCJA_%d', track_num);
        full_path = fullfile(output_dir, filename);
        save(full_path, 'DCJA_v', 'DCJA_pos', 'DCJA');
    end
    if  DCJA1 ~=1 && DSTNet1~=1
        VSIMM=DDHTA;
        VSIMM_pos=DDHTA_pos;
        VSIMM_v=DDHTA_v;
        output_dir = 'D:\Desktop\other1.3\paper1\testCode\CKF_IMM_VSIMM\figure';
        filename = sprintf('VSIMM_%d', track_num);
        full_path = fullfile(output_dir, filename);
        save(full_path, 'VSIMM_v', 'VSIMM_pos', 'VSIMM','meas_nosies_pos','meas_nosies_v');
    end
    if  DCJA1 ~=1&& DSTNet1==1
        DSTNet=DDHTA;
        DSTNet_pos=DDHTA_pos;
        DSTNet_v=DDHTA_v;
        output_dir = 'D:\Desktop\other1.3\paper1\testCode\CKF_IMM_VSIMM\figure';
        filename = sprintf('DSTNet_%d', track_num);
        full_path = fullfile(output_dir, filename);
        save(full_path, 'DSTNet_v', 'DSTNet_pos', 'DSTNet');
    end
    if  DCJA1 ==1&& DSTNet1==1
        output_dir = 'D:\Desktop\other1.3\paper1\testCode\CKF_IMM_VSIMM\figure';
        filename = sprintf('DDHTA_%d', track_num);
        full_path = fullfile(output_dir, filename);
        save(full_path, 'DDHTA_v', 'DDHTA_pos', 'DDHTA','meas_nosies_pos','meas_nosies_v');
    end
 else
     
 end


%Noise driven matrix
function G_total = get_G_matrix(dt, model_type)

    switch upper(model_type)
        case 'CV'
          
            G_single = [dt^2 / 2; dt;0];  
            dim_per_axis = 3;  
        case 'CA'
        
            G_single = [dt^3 / 6; dt^2 / 2; dt];  
            dim_per_axis = 3;  
        otherwise
            error('Invalid model type. Use ''CV'' or ''CA''.');
    end

    G_total = blkdiag(G_single, G_single, G_single);

    assert(size(G_total, 1) == 3 * dim_per_axis, 'Dimension mismatch.');
end
