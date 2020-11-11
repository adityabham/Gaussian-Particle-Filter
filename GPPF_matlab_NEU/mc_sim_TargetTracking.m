
clear all;
close all;

% Example taken from 
% 
% P. Closas, C. Fern�ndez-Prades, 2011, March. Particle filtering with adaptive number of particles. In 2011 Aerospace Conference (pp. 1-7). IEEE.

mc_runs = 10;
Nt = 100;
Np = 50;

var_prob = 0.0;

gp_verbose = 0;
% kernel function and gp config
o_learn_theta = false;       % if true theta acts as a theta0 and the gp will learn theta
reg = 1e-10;                 % (K + reg*I)

% theta = [log(1), log(1e-5)];    % kernel parameters
% kfcn = @(XN,XM,theta) (exp(theta(2))^2)*exp(-(pdist2(XN,XM).^2)/(2*exp(theta(1))^2)) + reg*(pdist2(XN,XM) == 0);

theta = [log(0.1)];    % kernel parameters
kfcn = @(XN,XM,theta) exp(-(pdist2(XN,XM).^2)/(2*exp(theta(1))^2)) + reg*(pdist2(XN,XM) == 0);

Ts = 1;
% x_t = F x_t-1 + Gamma * w_t 
F = [1 Ts 0 0;
     0 1 0 0;
     0 0 1 Ts;
     0 0 0 1];

Gamma = [Ts^2/2 0;
        Ts   0;
        0  Ts^2/2;
        0  Ts];
     
hfun = @(x) [30 - 10*log10(norm(-x(1:2:3))^2.2); atan2(x(3),x(1))];

% w_t ~ N (0, Q) 
q = sqrt(0.1);
Q = q^2 * eye(2);
Q4 = Gamma*Q*Gamma';
r = sqrt(0.001);
% r = 1;
% R = r^2 * eye(2);
R = [r^2 0; 0 r^2/10];

x0 = [100,0,100,0]';
P0 = diag([0.1, 0.01, 0.1, 0.01]); 


save_x = zeros(Nt,2);
save_y = zeros(Nt,2);

% save_kf_x = zeros(Nt,2);
% save_kf_y = zeros(Nt,2);

save_gppf_x_map = zeros(Nt, 2, mc_runs);
save_gppf_x_mmse = zeros(Nt,2, mc_runs);
save_gppf_Neff = zeros(Nt, mc_runs);

save_pf_x_map = zeros(Nt,2, mc_runs);
save_pf_x_mmse = zeros(Nt,2, mc_runs);
save_pf_Neff = zeros(Nt, mc_runs);


% kf = t_kf(x0, F, H, P0, Q4, R);

for r=1:mc_runs
    r
    x = x0;
    P = P0;

    particles = randn(Np,length(x0)) * sqrt(P0) + repmat(x0', Np,1);

    pf = t_pf(particles, @(z) F*z ,hfun, Q4, R);
    gppf = gp_pf(particles, @(z) F*z ,hfun, Q4, R, kfcn, theta, var_prob, o_learn_theta, gp_verbose);

    for n=1:Nt

%         n

        x = F *x + Gamma * mvnrnd([0, 0], Q)';
        y = hfun(x) + mvnrnd([0, 0], R)';

    %     kf.update(y);

        [MSE_x_pf, MAP_x_pf, Neff_pf] = pf.update(y);

        [MSE_x, MAP_x, Neff_gppf] = gppf.update(y);
    %     MSE_x = MSE_x_pf; MAP_x = MAP_x_pf; Neff_gppf = Neff_pf;

        save_x(n,:,r) = [x(1),  x(3)]';
        save_y(n,:,r) = y';

    %     save_kf_x(n,:) = [kf.x(1),  kf.x(3)]';
    %     save_kf_y(n,:) = y';

        save_pf_x_map(n,:,r) = [MAP_x_pf(1),  MAP_x_pf(3)]';
        save_pf_x_mmse(n,:,r) = [MSE_x_pf(1),  MSE_x_pf(3)]';
        save_pf_Neff(n,r) = Neff_pf;

        save_gppf_x_map(n,:,r) = [MAP_x(1),  MAP_x(3)]';
        save_gppf_x_mmse(n,:,r) = [MSE_x(1),  MSE_x(3)]';
        save_gppf_Neff(n,r) = Neff_gppf;

    end
 
end


% % % % % Plots

% figure;
% plot(save_x(:,1),save_x(:,2),'LineWidth',2), hold on, grid
% % % plot(save_kf_x(:,1),save_kf_x(:,2))
% plot(save_pf_x_map(:,1),save_pf_x_map(:,2),'-.','LineWidth',2)
% plot(save_pf_x_mmse(:,1),save_pf_x_mmse(:,2), '-.','LineWidth',2)
% plot(save_gppf_x_map(:,1),save_gppf_x_map(:,2), '--','LineWidth',2),
% plot(save_gppf_x_mmse(:,1),save_gppf_x_mmse(:,2), '--','LineWidth',2)
% % % legend('True','KF','GPPF (MAP)', 'GPPF (MMSE)', 'PF (MAP)', 'PF (MMSE)')
% xlabel('x [m]'), ylabel('y [m]')
% legend('True', 'PF (MAP)', 'PF (MMSE)', 'GPPF (MAP)', 'GPPF (MMSE)','Location','northwest')
% rectangle('Position',[-5 -5 10 10],'EdgeColor','r'), daspect([1 1 1])
% text(-5,12,'Sensor')

% RMSE_KF = sqrt((norm(save_kf_x - save_x).^2)/length(save_x))
for r=1:mc_runs
    RMSE_PF_MAP(r) = sqrt((norm(save_pf_x_map(:,:,r) - save_x(:,:,r)).^2)/Nt);
    RMSE_PF_MMSE(r) = sqrt((norm(save_pf_x_mmse(:,:,r) - save_x(:,:,r)).^2)/Nt);
    RMSE_GP_MAP(r) = sqrt((norm(save_gppf_x_map(:,:,r) - save_x(:,:,r)).^2)/Nt);
    RMSE_GP_MMSE(r) = sqrt((norm(save_gppf_x_mmse(:,:,r) - save_x(:,:,r)).^2)/Nt);
end
RMSE_PF_MAP = mean(RMSE_PF_MAP)
RMSE_PF_MMSE = mean(RMSE_PF_MMSE)
RMSE_GP_MAP = mean(RMSE_GP_MAP)
RMSE_GP_MMSE = mean(RMSE_GP_MMSE)

tvec = [0:Nt-1]*Ts;
figure;
plot(tvec, mean(sum(sqrt((save_pf_x_map - save_x).^2),2),3),'-.','LineWidth',2), hold on, grid
plot(tvec,mean(sum(sqrt((save_pf_x_mmse - save_x).^2), 2),3),'-.','LineWidth',2)
plot(tvec,mean(sum(sqrt((save_gppf_x_map - save_x).^2), 2),3),'--','LineWidth',2)
plot(tvec,mean(sum(sqrt((save_gppf_x_mmse - save_x).^2), 2),3),'--','LineWidth',2)
xlabel('time [s]'), ylabel('RMSE [m]')
legend('PF (MAP)', 'PF (MMSE)', 'GPPF (MAP)', 'GPPF (MMSE)')

figure,
plot(tvec,mean(save_pf_Neff,2),'-.','LineWidth',2), hold on, grid
plot(tvec,mean(save_gppf_Neff,2),'--','LineWidth',2)
xlabel('time [s]'), ylabel('Effective sample size')
axis([0 Nt 0 Np])
legend('PF', 'GPPF', 'location', 'best')

