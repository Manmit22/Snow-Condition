clc;
clear;
close all;
%%
file1 = double(imread('S:\Snow_contamination_condition\Rishi_Sir_corrected\Final_Documents\subset_2_of_collocateSentinel1_Landsat9_L2.tif'));
fileop = file1(:,:,8:13);
file = file1(:,:,1:7);
clear file1
clc;
%% Optical
clear optical norm_opt optical1
optical = reshape(fileop,[],6);
for i = 1:size(optical,2)
    norm_opt(:,i) = (optical(:,i)-min(optical(:,i)))./(max(optical(:,i))-min(optical(:,i)));
end
clear i
optical1 = reshape(norm_opt,size(file,1),size(file,2),6);
clear optical norm_opt
clc;
%%
clear SCI
SCI = (optical1(:,:,1)-optical1(:,:,2))./(optical1(:,:,1)+optical1(:,:,2));
SCI_res = reshape(SCI,[],1);

%%
clear NDSI NDIS
NDSI = (optical1(:,:,2)-optical1(:,:,5))./(optical1(:,:,2)+optical1(:,:,5));
%%
clear NDSI_res NDSI_snow f 
NDSI_res = reshape(NDSI,[],1);
f = find(NDSI_res<0.4);
NDSI_res(f,:) = NaN;
NDSI_snow = reshape(NDSI_res,size(NDSI,1),size(NDSI,2));
clear f
%%
clear lat lon f1 lon_f1 lat_f1
lat = round(reshape(file(:,:,4),[],1),4);
lon = round(reshape(file(:,:,5),[],1),4);

%%
f1 = find(lat~=28.6903);
lon_f1 = lon;
lon_f1(f1,:) = NaN;
clear f1
f1 = find(lon_f1==93.9678);
location1 = f1;
NDSI1 = NDSI_res(f1);
clear f1 lon_f1

f1 = find(lat~=28.7364);
lon_f1 = lon;
lon_f1(f1,:) = NaN;
clear f1
f1 = find(lon_f1==93.9656);
location2 = f1;
NDSI2 = NDSI_res(f1);
clear f1 lon_f1

f1 = find(lat~=28.7374);
lon_f1 = lon;
lon_f1(f1,:) = NaN;
f1 = find(lon_f1==93.9655);
location3 = f1;
NDSI3 = NDSI_res(f1);
clear f1 lon_f1

f1 = find(lat~=28.7381);
lon_f1 = lon;
lon_f1(f1,:) = NaN;
clear f1
f1 = find(lon_f1==93.9652);
location4 = f1;
NDSI4 = NDSI_res(f1);
clear f1 lon_f1

f1 = find(lat~=28.7387);
lon_f1 = lon;
lon_f1(f1,:) = NaN;
clear f1
f1 = find(lon_f1==93.9646);
location5 = f1;
NDSI5 = NDSI_res(f1);
clear f1 lon_f1

NDSI_field = [NDSI1;NDSI2;NDSI3;NDSI4;NDSI5];
locations = [location1;location2;location3;location4;location5];
% clear location1 location2 location3 location4 location5
clc;
clear ans NDSI1 NDSI2 NDSI3 NDSI4 NDSI5
%%
clear SD_field
SD_field  = [3;20;31;25;33;40;50;50]; %%% Snow Depth in cm observed in the field
% X = [1282;973;441;435;448;460;461;478];
% Y = [920;815;786;780;790;791;798;809];
clc;
%%
clear NDVI NDVFSI
NDFSI = (optical1(:,:,4)-optical1(:,:,5))./(optical1(:,:,4)+optical1(:,:,5));
NDVI = (optical1(:,:,4)-optical1(:,:,3))./(optical1(:,:,4)+optical1(:,:,3));
NDVI = (optical1(:,:,4)-optical1(:,:,3))./(optical1(:,:,4)+optical1(:,:,3));
%%
clear NDFSI_res NDFSI_snow
NDFSI_res = reshape(NDFSI,[],1);
NDVI_res = reshape(NDVI,[],1);
f = find(NDVI_res>0.6);
NDFSI_res(f,:) = NaN;
clear f
f = find(reshape(NDSI,[],1)>0.4);
NDFSI_res(f,:) = NaN;
clear f
f = find(NDFSI_res<=0.4);
NDFSI_res(f,:) = NaN;
clear f
NDFSI_snow = reshape(NDFSI_res,size(NDSI,1),size(NDSI,2));

clc;
%%
clear NDFSI_res1 NDSI_res1
f = find(~isnan(NDSI_res));
f1 = find(isnan(NDSI_res));
NDSI_res1 = NDSI_res;
NDSI_res1(f,:) = 1;
NDSI_res1(f1,:) = 0;
clear f f1
f = find(~isnan(NDFSI_res));
f1 = find(isnan(NDFSI_res));
NDFSI_res1 = NDFSI_res;
NDFSI_res1(f,:) = 1;
NDFSI_res1(f1,:) = 0;
clear f f1;
clc;
%% OR operator NDFSI
ESM = find((NDSI_res1|NDFSI_res1)==0);
New_SC = reshape(NDSI,[],1);
New_SC(ESM,:) = NaN;
New_SC_img = reshape(New_SC,size(NDSI,1),size(NDSI,2));
clc;
%%
clear NDSI_res NDFSI_res NDSI_field NDSI_res1 NDFSI_res1 NDFSI_snow ESM NDSI_snow NDVI_res ans
NIR = optical1(:,:,4);
Red = optical1(:,:,3);
SWIR = optical1(:,:,5);

Numo = (NIR.*(Red-SWIR));
Deno = ((NIR+SWIR).*(NIR+Red));

S3 = Numo./Deno;
clear NIR Red SWIR Numo Deno
clc;
%%
clear S3_res
S3_res = reshape(S3,[],1);
Pure_Snow_locations = find(S3_res<=0.18);
SUV1 = find(S3_res<0.0);
SUV2 = find(S3_res>0.18);
SUV_final = sort([SUV1;SUV2]);
S3_res(SUV_final,:) = NaN;
SUV = find(~isnan(SUV_final));
clear  SUV1 SUV2 SUV_final
clc;
%%
f = find(~isnan(New_SC));
f1 = find(isnan(New_SC));
New_SC1 = New_SC;
New_SC1(f,:) = 1;
New_SC1(f1,:) = 0;
clear f f1

f = find(~isnan(S3_res));
f1 = find(isnan(S3_res));
S3_res1 = S3_res;
S3_res1(f,:) = 1;
S3_res1(f1,:) = 0;
clear f f1 
%% OR operator ESM
ESM = find((New_SC1|S3_res1)==0);
New_ESM = reshape(NDSI,[],1);
New_ESM(ESM,:) = NaN;
New_ESM_img = reshape(New_ESM,size(NDSI,1),size(NDSI,2));
figure
imshow(New_ESM_img)
title('NDSI or NDFSI or S3')
clc;
clear S3_res New_SC ESM S3_res1 SUV New_SC1 New_SC New_SC_img
New_ESM(locations)
%%
f = find(isnan(New_ESM));
SCI_res(f,:) = NaN;
clear f f1
f = find(SCI_res<0);
f1 = find(SCI_res>=0);
Conta_snow = SCI_res;
Conta_snow(f1,:) = NaN;
Clear_snow = SCI_res;
Clear_snow(f,:) = NaN;
clear f f1;
clc;
%%
NDSI_norm = (NDSI+1)/2;
NDSI_norm_res = reshape(NDSI_norm,[],1);
NIR = optical1(:,:,4);
NIR_res = reshape(NIR,[],1);
figure
scatter(reshape(NDSI_norm,[],1),reshape(NIR,[],1))
xlabel('NDSI')
ylabel('NIR')
clc;
%%
clear w W
id = 0.385401;
iw = 0.00768211;
sd = (0.989914-id);
sw = (0.0039826-iw);
w = (id + (sd.*NDSI_norm_res)-NIR_res)./((id-iw) + ((sd-sw).*NDSI_norm_res));
W = w*10;
% find_Inf = find(W==Inf);
find_snow = find(isnan(New_ESM));
find_nega = find(W<0);
W(find_snow,:) = NaN;
W(find_nega,:) = NaN;
clc;
clear find_snow
%%

clear Radar Snow_sigma_vh Snow_sigma_vv LIA LOS Radar_conta Radar_clear
Radar1 = reshape(file(:,:,:),[],7);
LOS_shadow = find(Radar1(:,end)==1);
Radar1(LOS_shadow,:) = NaN;

Radar_vh = 10*log10(Radar1(:,1));
Radar_vv = 10*log10(Radar1(:,2));
clear Radar
Radar = [Radar_vh,Radar_vv,Radar1(:,3),Radar1(:,4),Radar1(:,5),Radar1(:,6)];


Radar_conta = Radar;
f = find(isnan(Conta_snow));
Radar_conta(f,:) = NaN;
f1 = find(isoutlier(Radar_conta(:,1))==1);
Radar_conta(f1,:) = NaN;


clear f f1
Radar_clear = Radar;
f = find(isnan(Clear_snow));
Radar_clear(f,:) = NaN;
f1 = find(isoutlier(Radar_clear(:,1))==1);
Radar_clear(f1,:) = NaN;

clear f f1
clc;
%%
clear W_conta W_clear
W_conta = W;
f = find(isnan(Conta_snow));
W_conta(f,:) = NaN;
clear f
W_clear = W;
f = find(isnan(Clear_snow));
W_clear(f,:) = NaN;
clear f
clc;
%%
clear a
f = find(~isnan(New_ESM));
f1 = find(isnan(New_ESM));

Snow_sigma_vh = Radar(:,1);
Snow_vh = Snow_sigma_vh;
Snow_vh(f1,:) = NaN;
TF1 = find(isoutlier(Snow_vh)==1);
Snow_vh(TF1,:) = NaN;
clear TF1
Non_Snow_vh = Snow_sigma_vh;
Non_Snow_vh(f,:) = NaN;
TF1 = find(isoutlier(Non_Snow_vh)==1);
Non_Snow_vh(TF1,:) = NaN;
clear TF1


Snow_sigma_vv = Radar(:,2);
Snow_vv = Snow_sigma_vv;
Snow_vv(f1,:) = NaN;
TF1 = find(isoutlier(Snow_vv)==1);
Snow_vv(TF1,:) = NaN;
clear TF1

Non_Snow_vv = Snow_sigma_vv;
Non_Snow_vv(f,:) = NaN;
TF1 = find(isoutlier(Non_Snow_vv)==1);
Non_Snow_vv(TF1,:) = NaN;
clear TF1

LIA = Radar(:,6);
Snow_LIA = LIA;
Snow_LIA(f1,:) = NaN;
Non_Snow_LIA = LIA;
Non_Snow_LIA(f,:) = NaN;

Z = Radar(:,3);
Snow_z = Z;
Snow_z(f1,:) = NaN;
Non_Snow_z = Z;
Non_Snow_z(f,:) = NaN;
clc;
clear f f1 ans LOS_shadow


%%
figure
subplot(1,4,1)
boxplot([((Snow_vh)),((Non_Snow_vh))],'Labels',{'Snow','Non Snow'})
ylabel('Backsacttering in dB: VH')
subplot(1,4,2)
boxplot([((Snow_vv)),((Non_Snow_vv))],'Labels',{'Snow','Non Snow'})
ylabel('Backsacttering in dB: VV')
subplot(1,4,3)
boxplot([Snow_LIA,Non_Snow_LIA],'Labels',{'Snow','Non Snow'})
ylabel('LIA')
subplot(1,4,4)
boxplot([Snow_z,Non_Snow_z],'Labels',{'Snow','Non Snow'})
ylabel('Elevation(m)')

clc;

%%
clear h p_w h_w
h_vh = ttest2(Snow_vh,Non_Snow_vh);
[~,h_w_vh] = ranksum(Snow_vh,Non_Snow_vh); 

h_vv = ttest2(Snow_vv,Non_Snow_vv);
[~,h_w_vv] = ranksum(Snow_vv,Non_Snow_vv); 

h_lia = ttest2(Snow_LIA,Non_Snow_LIA);
[~,h_w_lia] = ranksum(Snow_LIA,Non_Snow_LIA); 
clear p_w_vh p_w_vv
% 

% h_vh_vv = ttest2(Snow_vh_vv,Non_Snow_vh_vv);
% [~,h_w_vh_vv] = ranksum(Snow_vh_vv,Non_Snow_vh_vv); 
%% Dielectric estimation
clear h_vh h_w_vh h_vv h_w_vv id iw sd sw NDFSI S3
%%
clear Dry_snow_density_new


    dsw = W>=1;
    Dry_snow_wetness = W;
    Dry_snow_wetness(dsw,:) = NaN;
    Dry_snow_wetness_image = reshape(Dry_snow_wetness,size(NDSI,1),size(NDSI,2));
%    figure
%     imagesc(Dry_snow_wetness_image)
%     title('Dry Snow Wetness')
    
clear i
%%
clear dsw
for i = 1:length(W)
    dsd = Dry_snow_wetness(i)/5.35;
    c1 = -0.3205;
    if(isnan(dsd))
        Dry_snow_density_new(i,:) = NaN;
    else
        P = [1.861 0 c1 dsd];
        r = roots(P);
        Dry_snow_density_new(i,:) = abs(max(double(r)));
    end
end
 
clear i j eqn r P c1 dsd
clc;
%%
clear Dry_snow_dielectric Avg_ds_density Avg_ds_dielectric

    Dry_snow_dielectric = (1+(1.7*Dry_snow_density_new)+(0.7*((Dry_snow_density_new).^2)));
    Avg_ds_density = nanmean(Dry_snow_density_new);
    Avg_ds_dielectric = nanmean(Dry_snow_dielectric);

clear i
%% round upto second decimal  unique Wetness
clear Dry_snow_permittivity Gamma Beta
clear i
    unique_wetness = unique(W);
    Wetness_unique1 = round(unique_wetness,2);
    fnanw = isnan(Wetness_unique1);
    Wetness_unique1(fnanw,:) = [];
    permittivity = (1+(1.92*Avg_ds_density)+(Wetness_unique1/5.35));
    n = length(Wetness_unique1);
    sum_w1 = sum(Wetness_unique1);
    sum_w1_sq = sum((Wetness_unique1).^2);
    sum_w1_cub = sum((Wetness_unique1).^3);
    sum_w1_four = sum((Wetness_unique1).^4);
    sum_permi = sum(permittivity);
    sum_Pro_Per_Wet = sum((permittivity.*Wetness_unique1));
    sum_Pro_sq_Wet = sum(((permittivity.*((Wetness_unique1).^2))));
    syms beta gamma apha
    eqn1 = ((apha*n)+(sum_w1*beta)+(sum_w1_sq*gamma))== sum_permi;
    eqn2 = ((apha*sum_w1)+(sum_w1_sq*beta)+(sum_w1_cub*gamma))== sum_Pro_Per_Wet;
    eqn3 = ((apha*sum_w1_sq)+(sum_w1_cub*beta)+(sum_w1_four*gamma))== sum_Pro_sq_Wet;
    sol = solve([eqn1,eqn2,eqn3],[apha,beta,gamma]);
    Apha = double(sol.apha);
    Beta = double(sol.beta);
    Gamma = double(sol.gamma);
    Dry_snow_permittivity = Apha;
clear i
clc;
%% Snow Permittivity
clear sum_Pro_Per_Wet sum_Pro_sq_Wet sum_permi sum_w1_four sum_w1_cub sum_w1_sq
clear sum_w1 n permittivity Wetness_unique1 fnanw unique_wetness
clear beta gamma apha sol eqn1 eqn2 eqn3 sol Apha i

    Snow_Permittivity = ((Dry_snow_permittivity)+((Beta)*W)+(Gamma*((W).^2)));
clear i
clc;
%%
clear W_conta W_clear Snow_Permittivity_conta Snow_Permittivity_clear
Snow_Permittivity_conta = Snow_Permittivity;
Snow_Permittivity_clear = Snow_Permittivity;
W_conta = W;
f = find(isnan(Conta_snow));
W_conta(f,:) = NaN;
Snow_Permittivity_conta(f,:) = NaN;
clear f
W_clear = W;
f = find(isnan(Clear_snow));
W_clear(f,:) = NaN;
Snow_Permittivity_clear(f,:) = NaN;
clear f
clc;

%%
aa1 = Radar_conta(:,1);
bb1 = Radar_clear(:,1);
aa2 = Radar_conta(:,2);
bb2 = Radar_clear(:,2);
clc;
%%
figure
subplot(1,6,1)
boxplot([aa1,bb1],'Labels',{'Contaminated','Clear'})
ylabel('Backscattering in dB VH')
subplot(1,6,2)
boxplot([aa2,bb2],'Labels',{'Contaminated','Clear'})
ylabel('Backscattering in dB VV')
subplot(1,6,3)
boxplot([((Radar_conta(:,6))),((Radar_clear(:,6)))],'Labels',{'Contaminated','Clear'})
ylabel('LIA')
subplot(1,6,4)
boxplot([((Radar_conta(:,3))),((Radar_clear(:,3)))],'Labels',{'Contaminated','Clear'})
ylabel('Elevation (m)')
subplot(1,6,5)
boxplot([W_conta,W_clear],'Labels',{'Contaminated','Clear'})
ylabel('Wetness')
subplot(1,6,6)
boxplot([Snow_Permittivity_conta,Snow_Permittivity_clear], 'Labels' ,{'Contaminated','Clear'})
ylabel('Permittivity')
clc;
%%
clear Snow_Img RGB
RGB(:,:,1) = optical1(:,:,3);
RGB(:,:,2) = optical1(:,:,2);
RGB(:,:,3) = optical1(:,:,1);
% figure
% imshow(RGB,[])
% title('RGB')
clc;
%%
clear Snow_Img_cont Snow_Img_clear
Snow_Img_cont = New_ESM;
Snow_Img_clear = New_ESM;
f = find(~isnan(New_ESM));
Snow_percentage = (length(f)/length(New_ESM))*100;
clear f
f = find(isnan(Conta_snow));
f1 = find(~isnan(Conta_snow));
Conta_per = (length(f1)/length(New_ESM))*100;
Snow_Img_cont(f,:) = 0;
clear f f1

f = find(isnan(Clear_snow));
f1 = find(~isnan(Clear_snow));
Clear_per = (length(f1)/length(New_ESM))*100;
Snow_Img_clear(f,:) = 0;
clear f f1

Conta_Snow_img = reshape(Snow_Img_cont,size(NDSI,1),size(NDSI,2));
Clear_Snow_img = reshape(Snow_Img_clear,size(NDSI,1),size(NDSI,2));

BW_conta = imoverlay(RGB,Conta_Snow_img,'blue');
BW_clear = imoverlay(RGB,Clear_Snow_img,'blue');

figure
subplot(1,4,1)
imshow(RGB)
title('RGB')
subplot(1,4,2)
imshow(New_ESM_img)
title('Snow Cover')
subplot(1,4,3)
imshow(BW_conta)
title('Contaminated snow')
subplot(1,4,4)
imshow(BW_clear,[])
title('Clear Snow')
clc;

%% Scatter plot between SAR backscattering and Local Incidence Angle
figure
subplot(1,2,1)
scatter((Radar_conta(:,6)),(Radar_conta(:,1)))
xlabel('LIA in degrees')
xlim([20 60])
ylabel('VH (dB)')
subplot(1,2,2)
scatter((Radar_conta(:,6)),(Radar_conta(:,2)))
xlabel('LIA in degrees')
xlim([20 60])
ylabel('VV (dB)')
clc;
%% Scatter plot between SAR backscattering and Snow contamination Index
figure
subplot(1,2,1)
scatter(SCI_res,(Radar_conta(:,1)))
xlabel('SCI')
ylabel('VH (dB)')
subplot(1,2,2)
scatter(SCI_res,(Radar_conta(:,2)))
xlabel('SCI')
ylabel('VV (dB)')
clc;