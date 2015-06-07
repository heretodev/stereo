% Alg 12.1 Optimal Triangulation Method (Hartley & Zisserman)
function X = triangulateCorrespondences(center, x1, x2, F, P1, P2)
    [x1, x2] = correctCorrespondences(center, x1, x2, F);
    % Get 3x4 project P1, P2
    P1 = eye(3,4);
    [U,D,V] = svd(F,0);
	ep2 = U(:,3);
	P2 = skew(ep2)' * F;
    P2 = [P2, ep2];
    
    % 12.2 (Hartley & Zisserman): Use DLT: From x1 (cross) P1 * X = 0 
    % i.e. x1 cross x1 = 0, where x1 = P1 * X.
    X = [];
    for i = 1 : size(x1, 2)
        A = [ x1(1, i) .* P1(3,:) - P1(1,:)
              x1(2, i) .* P1(3,:) - P1(2,:)
              x2(1, i) .* P2(3,:) - P2(1,:)
              x2(2, i) .* P2(3,:) - P2(2,:) ];
        % Solve for X, AX = 0
        [U, D, V] = svd(A, 0);
        X = [X, V(:, 3) ./ V(4, 3)];
    end
    
    figure
    scatter3(X(1,:),X(2,:),X(3,:),'filled');
end
% For computation of P2 - investigate later
function S_hat = skew(S)
    S_hat = zeros(3,3);
    S_hat(1,2) = - S(3); S_hat(1,3) = S(2);   S_hat(2,3) = - S(1);
    S_hat(2,1) = S(3);   S_hat(3,1) = - S(2); S_hat(3,2) = S(1);
end

% Alg 12.1 Hartley, Optimal Triangulation Method, numerically verified
% identical to OpenCV's, excluding its roots error.
function [x1, x2] = correctCorrespondences(center, x1, x2, F)
    F0 = F;
    for i = 1 : size(x1, 2)
        % i. translate x1 and x2 to origin 0,0
        T1 = [1 0 x1(1, i)
              0 1 x1(2, i)
              0 0 1         ];
        T2 = [1 0 x2(1, i)
              0 1 x2(2, i)
              0 0 1         ];
        F = T2' * F0 * T1; % ii

        % Find epipoles of translated F
        [U,D,V] = svd(F,0);
        ep1 = V(:,3);
        ep2 = U(:,3);

        % iii. Normalize epipoles so ep1(1)^2 + ep1(1)^2 = 1
        denom = (ep1(1) * ep1(1) + ep1(2) * ep1(2));
        ep1 = ep1 ./ denom;
        if(ep1(3) < 0)
            ep1 = -ep1;
        end
        denom = (ep2(1) * ep2(1) + ep2(2) * ep2(2));
        ep2 = ep2 ./ denom;
        if(ep2(3) < 0)
            ep2 = -ep2;
        end

        R1 = [ ep1(1) ep1(2) 0
              -ep1(2) ep1(1) 0
              0       0     1];    
        R2 = [ ep2(1) ep2(2) 0
              -ep2(2) ep2(1) 0
              0       0     1];
        F = R2 * F * R1'; % v

        f1 = ep1(3); f2 = ep2(3);
        a = F(2,2); b = F(2,3); c = F(3,2); d = F(3,3);

        % vii: g(t) = t((at + b)^2 + f2^2(ct + d)^2)^2 -(ad - bc)(1 + f1^2t^2)^2(at + b)(ct + d)
        g = [-a*d*d*b+b*b*c*d, ...
            +f2*f2*f2*f2*d*d*d*d+b*b*b*b+2*b*b*f2*f2*d*d-a*a*d*d+b*b*c*c, ...
            +4*a*b*b*b+4*b*b*f2*f2*c*d+4*f2*f2*f2*f2*c*d*d*d-a*a*d*c+b*c*c*a+4*a*b*f2*f2*d*d-2*a*d*d*f1*f1*b+2*b*b*c*f1*f1*d, ...
            +6*a*a*b*b+6*f2*f2*f2*f2*c*c*d*d+2*b*b*f2*f2*c*c+2*a*a*f2*f2*d*d-2*a*a*d*d*f1*f1+2*b*b*c*c*f1*f1+8*a*b*f2*f2*c*d, ...
            +4*a*a*a*b+2*b*c*c*f1*f1*a+4*f2*f2*f2*f2*c*c*c*d+4*a*b*f2*f2*c*c+4*a*a*f2*f2*c*d-2*a*a*d*f1*f1*c-a*d*d*f1*f1*f1*f1*b+b*b*c*f1*f1*f1*f1*d, ...
            +f2*f2*f2*f2*c*c*c*c+2*a*a*f2*f2*c*c-a*a*d*d*f1*f1*f1*f1+b*b*c*c*f1*f1*f1*f1+a*a*a*a, ...
            +b*c*c*f1*f1*f1*f1*a-a*a*d*f1*f1*f1*f1*c];

        % solve for t to get up to 6 real roots
        t = roots(fliplr(g));
        tinf = 1. / (f1*f1) + (c*c) / (a*a + f2*f2*c*c);
        
        % Get the real components of the real roots.
        t = real(t(find(arrayfun(@(x) imag(x) == 0, t))));
        
        % viii: select tm, the minimum of evaluated cost of each t
        t = [t; tinf];
        gt = (t.*t) ./ (1 + f1.*f1.*t.*t) + ((c.*t + d).*(c.*t + d)) ./ ((a.*t + b).*(a.*t + b) + f2.*f2.*(c.*t + d).*(c.*t + d));
        [m, mi] = min(gt);
        tm = t(mi);

        % ix:
        %l1 = [tm * f1, 1, -tm]';
        %l2 = [-f2(ctm + d), atm + b, ctm + d]';
        % find x1 and x2 at the closest points on these lines to the origin.
        % for line [lambda, mu, v], the closest point to the orgin is
        % [-lambda * v, -mu * v, lambda * lambda + mu * mu]';
        x1h = [tm*tm*f1, ...
               tm, ...
               tm*tm*f1*f1 + 1]';
        x1h = x1h ./ x1h(3);

        x2h = [ f2*(c*tm + d).*(c*tm + d), ...
                -(a*tm + b).*(c*tm + d), ...
                f2*f2*(c*tm + d).*(c*tm + d) + (a*tm + b).*(a*tm + b)]';
        x2h = x2h ./ x2h(3);
        

        % x:
        x1(:, i) = T1 * R1' * x1h;
        x2(:, i) = T2 * R2' * x2h;
    end
end
