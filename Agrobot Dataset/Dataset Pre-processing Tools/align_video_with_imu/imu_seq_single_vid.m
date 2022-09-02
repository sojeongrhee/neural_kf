%% import and sequence
inner_split = 0; %for those files where the log is not continuous
lima = 8002; limb = 36902;
if(inner_split == 1)
    lima_int = 10099;
    limb_int = 16699;
end
z1 = readtable('csv/log 7/raw_imu_gps_2022-05-04-01-46-38.csv');
z2 = readtable('csv/log 7/mag_imu_gps_2022-05-04-01-46-38.csv');
z3 = readtable('csv/log 7/data_imu_gps_2022-05-04-01-46-38.csv');
v1 = load('videos/8/DJI_0008all.mat');
filename = 'log7_aligned.csv';
z1.x_time = (z1.x_time - min(z1.x_time))/1e9;
z2.x_time = (z2.x_time - min(z2.x_time))/1e9;
z3.x_time = (z3.x_time - min(z3.x_time))/1e9;
z1.x_time = z1.x_time-z1.x_time(1);
z2.x_time = z2.x_time-z2.x_time(1);
z3.x_time = z3.x_time-z3.x_time(1);
if(inner_split == 1)
    z1_a = z1(lima:lima_int,:);
    z2_a = z2(lima:lima_int,:);
    z3_a = z3(lima:lima_int,:);
    z1_b = z1(limb_int:limb,:);
    z2_b = z2(limb_int:limb,:);
    z3_b = z3(limb_int:limb,:);    
    z1 = vertcat(z1_a,z1_b);
    z2 = vertcat(z2_a,z2_b);
    z3 = vertcat(z3_a,z3_b);
else
    z1 = z1(lima:limb,:);
    z2 = z2(lima:limb,:);
    z3 = z3(lima:limb,:);
end


%% remove unnecessary variables
z1 = removevars(z1,{'field_header_seq','field_header_stamp', ...
    'field_header_frame_id','field_orientation_x', ...
    'field_orientation_y', 'field_orientation_z', 'field_orientation_w', ...
    'field_angular_velocity_x','field_angular_velocity_y','field_angular_velocity_z',...   
    'field_orientation_covariance0','field_orientation_covariance1', ...
    'field_orientation_covariance2','field_orientation_covariance3', ...
    'field_orientation_covariance4','field_orientation_covariance5', ...
    'field_orientation_covariance6','field_orientation_covariance7', ...
    'field_orientation_covariance8', ...
    'field_angular_velocity_covariance0', 'field_angular_velocity_covariance1', ...
    'field_angular_velocity_covariance2', 'field_angular_velocity_covariance3', ...
    'field_angular_velocity_covariance4', 'field_angular_velocity_covariance5', ...
    'field_angular_velocity_covariance6', 'field_angular_velocity_covariance7', ...
    'field_angular_velocity_covariance8', ...
    'field_linear_acceleration_covariance0', 'field_linear_acceleration_covariance1',...
    'field_linear_acceleration_covariance2', 'field_linear_acceleration_covariance3',...
    'field_linear_acceleration_covariance4', 'field_linear_acceleration_covariance5',...
    'field_linear_acceleration_covariance6', 'field_linear_acceleration_covariance7',...
    'field_linear_acceleration_covariance8'});

z2 = removevars(z2,{'x_time','field_header_seq','field_header_stamp', ...
    'field_header_frame_id','field_magnetic_field_covariance0',...
    'field_magnetic_field_covariance1', 'field_magnetic_field_covariance2', ...
    'field_magnetic_field_covariance3', 'field_magnetic_field_covariance4', ...
    'field_magnetic_field_covariance5', 'field_magnetic_field_covariance6', ...
    'field_magnetic_field_covariance7', 'field_magnetic_field_covariance8'});

z3 = removevars(z3,{'x_time','field_header_seq','field_header_stamp', ...
    'field_header_frame_id', ...
    'field_orientation_covariance0','field_orientation_covariance1', ...
    'field_orientation_covariance2','field_orientation_covariance3', ...
    'field_orientation_covariance4','field_orientation_covariance5', ...
    'field_orientation_covariance6','field_orientation_covariance7', ...
    'field_orientation_covariance8', ...
    'field_angular_velocity_covariance0', 'field_angular_velocity_covariance1', ...
    'field_angular_velocity_covariance2', 'field_angular_velocity_covariance3', ...
    'field_angular_velocity_covariance4', 'field_angular_velocity_covariance5', ...
    'field_angular_velocity_covariance6', 'field_angular_velocity_covariance7', ...
    'field_angular_velocity_covariance8', ...
    'field_linear_acceleration_covariance0', 'field_linear_acceleration_covariance1',...
    'field_linear_acceleration_covariance2', 'field_linear_acceleration_covariance3',...
    'field_linear_acceleration_covariance4', 'field_linear_acceleration_covariance5',...
    'field_linear_acceleration_covariance6', 'field_linear_acceleration_covariance7',...
    'field_linear_acceleration_covariance8'});
z1 = renamevars(z1,["field_linear_acceleration_x","field_linear_acceleration_y",...
    "field_linear_acceleration_z"],["field_linear_acceleration_x_RAW", ...
    "field_linear_acceleration_y_RAW","field_linear_acceleration_z_RAW"]);
imu_file = horzcat(z1,z2,z3);
clear z1 z2 z3;
rob_x_master = v1.rob_x';
rob_y_master = v1.rob_y';

%% upsample ground truth and create single table

rob_x_master = interp1(linspace(0,1,length(rob_x_master)), rob_x_master, (linspace(0,1,height(imu_file))))';
rob_y_master = interp1(linspace(0,1,length(rob_y_master)), rob_y_master, (linspace(0,1,height(imu_file))))';
master_table = horzcat(imu_file,table(rob_x_master,rob_y_master,'VariableNames', {'X','Y'}));

clearvars -except master_table filename

writetable(master_table,filename);