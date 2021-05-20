
close all
clear all
clc

% Cavity parameters

lambda0=500e-9;            %% Central wavelength design [m]
na = 3.0;                     %% DBR refractive index-a, AlAs
nb = 3.6;                   %% DBR refractive index-b, GaAs
nc = 3.6;                   %% refractive index of the cavity, GaAs
lc = 2 * lambda0/(nc);    %% Lenght of the cavity [m]
LQW= 10e-9;                 %% quantum well thickness in which the gain will be [m]
N_DBRn=28;                  %% amount of DBR n-doped pairs
N_DBRp=21;                  %% amount of DBR p-doped pairs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lambda_vec=linspace(400,600,1200)*1e-9;            %% Wavelength [m]
lambda_vec=sort([lambda_vec lambda0]);              %% here, I make sure lambda0 is inside the vector lambda

Gain=[0:10:2000]*1e2;                               %% Gain [m-1]

for jj=1:length(lambda_vec)

    lambda=lambda_vec(jj);
    [T,R]=Transmission_VCSEL_f(lambda,0,lambda0,na,nb,nc,N_DBRn,N_DBRp,lc,LQW);
    
    Trans(jj,:)=T;
    Reflc(jj,:)=R;
end

[T,R]=Transmission_VCSEL_f(lambda0,Gain,lambda0,na,nb,nc,N_DBRn,N_DBRp,lc,LQW);

idx_T = find( T==max(T) );
Gth   = Gain(idx_T);
LambdaGain=[lambda0 Gth max(T)];
display(strcat('lambda=',num2str(lambda0*1e6),'um ; ThGain=',num2str(Gth/100),'cm-1'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%X0fig=-3500; Y0fig=100;
X0fig=100; Y0fig=100;
Wfig=1000;Hfig=800;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



figure('Name','Results','position',[X0fig Y0fig Wfig Hfig])
subplot(1,1,1)
hold on;grid on;
xscale=[lambda_vec(1) lambda_vec(end)]*1e6;
yscale1=[0 1];
yscale2=[0 LambdaGain(2)/100*1.5];
[AX,H1,H2] = plotyy(lambda_vec*1e6,Trans,LambdaGain(:,1)*1e6,LambdaGain(:,2)/100);
set(H1,'color','b','linewidth',1,'marker','none');
set(H2,'color','r','linestyle','none','marker','o');
set(AX(1),'ycolor','b','xlim',xscale,'ylim',yscale1,'ytick',[0:0.1:1],'fontsize',15);
set(AX(2),'ycolor','r','xlim',xscale,'ylim',yscale2,'ytick',0:200:LambdaGain(2)/10,'fontsize',15);
xlabel('lambda (um)')
ylabel(AX(1),'Transmission')
ylabel(AX(2),'Threshold Gain (cm-1)')
title(strcat('\lambda0=',num2str(lambda0*1e9),'nm'))

figure
subplot(1,1,1,'fontsize',15)

semilogy(Gain/100,T,'r.-')
hold on; grid on;
ylim([1e-1 1e6])


semilogy(Gain/100,R,'y.-')
legend('Transimisson','Reflectance')
ylabel('Transmission','fontsize',13)
xlabel('Gain (cm-1)','fontsize',13)
title(strcat('\fontsize{15}ThresholdGain',' @\lambda=',num2str(lambda0*1e9),'nm'))
ylim([1e-1 1e6])

