clear; clc; close all;

%% Load UR5 robot model
ur5 = loadrobot("universalUR5");
ur5.DataFormat = "row";

%% Joint angles in degrees %% Test 2 are the angles after
theta1 = 140; %%  -60    
theta2 = -70; %% -130
theta3 = 70;  %%  120
theta4 = -30; %%  -70
theta5 = 40;  %%   70
theta6 = 50;  %% -140

q = deg2rad([theta1 theta2 theta3 theta4 theta5 theta6]);

%% Display robot
figure('Color','w');
ax = axes;
show(ur5, q, 'Parent', ax, 'Frames', 'off', 'PreservePlot', false);
view(145, 25);
axis equal;
grid on;
title('UR5 Forward Kinematics Check');
xlabel('X');
ylabel('Y');
zlabel('Z');
camlight headlight;
lightangle(45, 30);
material dull;

%% Get end-effector pose
T06 = getTransform(ur5, q, "tool0");

disp('End-effector transform T06:');
disp(T06);

%% Extract position
pos = T06(1:3,4);
fprintf('Position:\n');
fprintf('X = %.4f mm\n', pos(1)*1000);
fprintf('Y = %.4f mm\n', pos(2)*1000);
fprintf('Z = %.4f mm\n', pos(3)*1000);

%% Optional: show a marker at the tool point
hold on;
plot3(pos(1), pos(2), pos(3), 'ro', 'MarkerSize', 8, 'LineWidth', 2);