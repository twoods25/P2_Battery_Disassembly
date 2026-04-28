


%%
% Definer thetas som symbolske variabler (fjern dette, hvis de allerede har numeriske værdier)
theta1 = 10;
theta2 = 10;
theta3 = 10;
theta4 = 10;
theta5 = 10;
theta6 = 10;

% --- T01 ---
T01 = [ cosd(theta1), -sind(theta1), 0,    0;
        sind(theta1),  cosd(theta1), 0,    0;
                   0,             0, 1, 89.2;
                   0,             0, 0,    1];

% --- T12 ---
T12 = [-cosd(theta2),  sind(theta2),  0, 0;
                   0,             0, -1, 0;
       -sind(theta2), -cosd(theta2),  0, 0;
                   0,             0,  0, 1];

% --- T23 ---
T23 = [ cosd(theta3), -sind(theta3), 0, 425;
        sind(theta3),  cosd(theta3), 0,   0;
                   0,             0, 1,   0;
                   0,             0, 0,   1];

% --- T34 ---
T34 = [ cosd(theta4), -sind(theta4), 0,   392;
        sind(theta4),  cosd(theta4), 0,     0;
                   0,             0, 1, 109.3;
                   0,             0, 0,     1];

% --- T45 ---
T45 = [ cosd(theta5), -sind(theta5), 0,     0;
                   0,             0, 1, 94.75;
       -sind(theta5), -cosd(theta5), 0,     0;
                   0,             0, 0,     1];

% --- T56 ---
T56 = [-cosd(theta6),  sind(theta6),  0,     0;
                   0,             0, -1, -82.5;
       -sind(theta6), -cosd(theta6),  0,     0;
                   0,             0,  0,     1];
%theta values assigned

%total transformation
T06_trans = [     0.182539,     0.660858,     0.727974,   800.000000 ;
     -0.962690,     0.270573,    -0.004234,   280.000000 ;
     -0.199768,    -0.700040,     0.685592,   260.000000 ;
      0.000000,     0.000000,     0.000000,     1.000000 ];

%%Or test 1:
%T06_trans = [     0.643705,    -0.566804,     0.514176,   500.000000 ;
%      0.652314,     0.757720,     0.018632,   500.000000 ;
%     -0.400162,     0.323411,     0.857482,   500.000000 ;
%      0.000000,     0.000000,     0.000000,     1.000000 ];


fprintf("T06 using transformation matricies = \n")
disp(T06_trans)

alpha = deg2rad([0, 90, 0, 0, -90, 90]);
a = [0, 0, 425, 392, 0, 0];
d = [89.2, 0, 0, 109.3, 94.75, 82.5];
theta = deg2rad([0, 0 + 180, 0, 0, 0, 0 + 180]);


%% Inverse Kinematics for UR5
disp('Claude Cleanup')

a2 = -a(3);  % 425 mm
a3 = -a(4);  % 392 mm

% ─── Theta 1 ────────────────────────────────────────────────────────────────
P65   = [0; 0; -d(6); 1];
P05   = T06_trans * P65;
phi_1 = atan2d(P05(2), P05(1));
phi_2 = [1, -1] * acosd(d(4) / sqrt(P05(1)^2 + P05(2)^2));
theta_1 = phi_1 + phi_2 + 90;
%fprintf('theta_1: [%.4f, %.4f] grader\n\n', theta_1(1), theta_1(2));

% ─── Theta 5 ────────────────────────────────────────────────────────────────
r_13    = T06_trans(1,3);
r_23    = T06_trans(2,3);
theta_5 = zeros(2, 2);
for i = 1:2
    theta_5(i,:) = [1,-1] * acosd(sind(theta_1(i))*r_13 - cosd(theta_1(i))*r_23);
end
%fprintf('theta_5 løsninger:\n'); disp(theta_5);

% ─── Theta 6 ────────────────────────────────────────────────────────────────
T60   = inv(T06_trans);
X60_x = T60(1,1);  Y60_x = T60(1,2);
X60_y = T60(2,1);  Y60_y = T60(2,2);
theta_6 = zeros(2, 2);
for i = 1:2
    t1      = theta_1(i);
    sin_num = -X60_y * sind(t1) + Y60_y * cosd(t1);
    cos_num =  X60_x * sind(t1) - Y60_x * cosd(t1);
    for j = 1:2
        t5 = theta_5(i,j);
        if abs(sind(t5)) < 1e-10
            theta_6(i,j) = 0;  % Singularitet: theta_6 sættes til 0
        else
            theta_6(i,j) = atan2d(sin_num/sind(t5), cos_num/sind(t5));
        end
    end
end
%fprintf('theta_6 løsninger:\n'); disp(theta_6);

% ─── Theta 3, 2 og 4 (cobined in one loop) ──────────────────────────────────────
theta_3 = nan(2, 2, 2);
theta_2 = nan(2, 2, 2);
theta_4 = nan(2, 2, 2);

fprintf('\n--- Alle IK-konfigurationer [t1, t2, t3, t4, t5, t6] ---\n');
cfg_nr = 0;

for i = 1:2
    t1   = theta_1(i);
    T01c = [ cosd(t1), -sind(t1), 0,    0;
             sind(t1),  cosd(t1), 0,    0;
                    0,         0, 1, 89.2;
                    0,         0, 0,    1];

    for j = 1:2
        t5   = theta_5(i,j);
        t6   = theta_6(i,j);

        % Byg T46 én gang – genbruges til theta_3 OG theta_4
        T45c = [ cosd(t5), -sind(t5), 0,     0;
                        0,         0, 1, 94.75;
                -sind(t5), -cosd(t5), 0,     0;
                        0,         0, 0,     1];
        T56c = [-cosd(t6),  sind(t6),  0,     0;
                        0,         0, -1, -82.5;
                -sind(t6), -cosd(t6),  0,     0;
                        0,         0,  0,     1];
        T46c = T45c * T56c;

        % Calculate T14 once for both theta 2 and 3
        T14c = inv(T01c) * T06_trans * inv(T46c);
        p14x   = T14c(1,4);
        p14z   = T14c(3,4);
        len_P14xz   = sqrt(p14x^2 + p14z^2);

        % Theta 3: Law of cosine
        current_t3 = (len_P14xz^2 - a2^2 - a3^2) / (2*a2*a3);
        if current_t3 > 1.0001 || current_t3 < -1.0001
            fprintf('Konfiguration (t1=%.2f, t5=%.2f): Punkt ikke inden for rækkevidde!\n', t1, t5);
            continue;
        end
        current_t3      = max(-1, min(1, current_t3));
        t3_vals  = [acosd(current_t3), -acosd(current_t3)];

        for k = 1:2
            t3 = t3_vals(k);
            theta_3(i,j,k) = t3;

            % Theta 2: atan2 and asin
            phi_1_t2        = atan2d(-p14z, -p14x);
            arcsin_arg      = (-a3 * sind(t3)) / len_P14xz;
            phi_2_t2        = asind(arcsin_arg);
            t2              = phi_1_t2 - phi_2_t2;
            theta_2(i,j,k)  = t2;

            % Theta 4: isolate T34 and extract theta 4 using atan2
            T12c = [-cosd(t2),  sind(t2),  0, 0;
                            0,         0, -1, 0;
                    -sind(t2), -cosd(t2),  0, 0;
                            0,         0,  0, 1];
            T23c = [ cosd(t3), -sind(t3), 0, 425;
                     sind(t3),  cosd(t3), 0,   0;
                            0,         0, 1,   0;
                            0,         0, 0,   1];
            T34c            = inv(T12c * T23c) * T14c;
            t4              = atan2d(T34c(2,1), T34c(1,1));
            theta_4(i,j,k)  = t4;

            % Print configuration
            cfg_nr = cfg_nr + 1;
            fprintf('(%2d)-[%8.2f°, %8.2f°, %8.2f°, %8.2f°, %8.2f°, %8.2f°]\n', ...
                cfg_nr, t1, t2, t3, t4, t5, t6);
        end
    end
end

