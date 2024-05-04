function [FBOp, BWOp] = util_syn_meas_op_single(A, At, G, W, nWimag, flag_complex)

if exist('nWimag', 'var') && ~isempty(nWimag)
    FBOp = @(x) forward_operator_precond_G(x, G, W, A, nWimag);
    if exist('flag_complex', 'var') && flag_complex
        BWOp = @(y) adjoint_operator_precond_G(y, G, W, At, nWimag);
    else
        BWOp = @(y) real(adjoint_operator_precond_G(y, G, W, At, nWimag));
    end
else
    FBOp = @(x) forward_operator_G(x, G, W, A);
    if exist('flag_complex', 'var') && flag_complex
        BWOp = @(y) adjoint_operator_G(y, G, W, At);
    else
        BWOp = @(y) real(adjoint_operator_G(y, G, W, At));
    end
end

end

function y = forward_operator_G(x, G, W, A)
Fx = A(x);
y = G * Fx(W);
end

function x = adjoint_operator_G(y, G, W, At)
g2 = zeros(size(W, 1), 1);
g2(W) = G' * y;
x = At(g2);
end

function y = forward_operator_precond_G(x, G, W, A, aW)
Fx = A(x);
y = sqrt(aW) .* (G * Fx(W));
end

function x = adjoint_operator_precond_G(y, G, W, At, aW)
g2 = zeros(size(W, 1), 1);
g2(W) = G' * (sqrt(aW) .* y);
x = At(g2);
end