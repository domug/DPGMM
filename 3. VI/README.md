




$$
\begin{aligned}



&\textbf{<Parameters>} \\[5pt]

\text{latent parameters: }& \textbf{W} = \{ (V_i)_i, (\eta^*_i)_i, (Z_n)_n \} \\[5pt]

\text{observations: }& \textbf{x} = (x_n)_n  \\[5pt]

\text{mixture proportion: }& V_i \sim Beta(1,\alpha) \\[5pt]

\text{stick length: }& \pi_i(\textbf{V}) = V_i \prod_{j=1}^{i-1} (1-V_j) \quad \dots \text{(SBP representation)}\\[5pt]

\text{mixture components: }& Z_n | \pi \sim Cat(\pi) \\[5pt]

\text{mixture atoms: }& \eta_i^* \sim G_0 \quad \Rightarrow p(\eta_t^* | \lambda) = h(\eta_t^*)exp(\lambda_1^\intercal - \lambda_2 a(\eta_t^*) - a(\lambda)) \quad \dots (G_0\text{ belonging to exponential family)} \\[5pt]

\Rightarrow \quad& G = \sum_{i=1}^\infty \pi_i(\textbf{V})\delta_{\eta_i^*} \sim DP(\alpha, G_0)

\\[40pt]



&\textbf{<KL Divergence>} \\[5pt]

\text{minimize}:\quad D(q_v || p(.|\textbf{x}, \alpha, \lambda)) &= E_q[logq_v(\textbf{W})] - E_q[logp(\textbf{W}, \textbf{x} |\alpha, \lambda)] + logp(\textbf{x} | \textbf{W}, \alpha, \lambda) \geq 0 \\[5pt]

\text{is equivalent to maximize}:\quad logp(\textbf{x}|\alpha,\lambda) &\geq E_q[logp(\textbf{W}, \textbf{x} | \alpha, \lambda)] - E_q[logq_v(\textbf{W})] \quad \dots \textbf{(ELBO)} 

\\[40pt]



&\textbf{<ELBO for DPMM>} \\

log p(x|\alpha, \lambda) &\geq E_q[ logp(V|\alpha)] + E_q[logp(\eta^*|\lambda)] + \sum_{n=1}^N \big(E_q[logp(Z_n|V)] \big) + E_q[logp(x_n|Z_n)] \\

&\quad - E_q[logq_v(V,\eta^*,Z)], \\[10pt]

\text{where }\quad &q_v(V,\eta^*,Z) \approx \prod_{t=1}^{T-1} q_{\gamma_t}(V_t) \prod_{t=1}^{T} q_{\tau_t}(\eta_t^*) \prod_{n=1}^{N} q_{\phi_n}(Z_n) \quad \dots \text{(mean-field approximation)} \\[15pt]

&\Leftrightarrow E_q[logq_v(V,\eta^*,Z)] = \sum_{t=1}^{T-1}  E_q[q_{\gamma_t}(V_t)] + \sum_{t=1}^{T-1}  E_q[q_{\tau_t}(\eta_t^*)] + \sum_{n=1}^{N}  E_q[q_{\phi_n}(Z_n)]

\\[40pt]



&\textbf{<Calculations>} \\

E_q[ logp(V|\alpha)] &= \sum_{t=1}^{T-1} (\alpha-1) E_q[log(1-V_t)] - (T-1)(log\Gamma(\alpha) - log\Gamma(1+\alpha)) \\

E_q[logp(\eta^*|\lambda)] &= \sum_{t=1}^T \big(logh(\eta_t^*) +\lambda_1^\intercal E_q[\eta_t^*] - \lambda_2 E_q[a(\eta_t^*)] -T a(\lambda)\big) \\

E_q[logp(Z_n|V)] &= \sum_{i=1}^T q(z_n=t) E_q[logV_t] + q(z_n > t) E_q[log(1-V_t)] \\

E_q[logp(x_n|Z_n)] &= \sum_{t=1}^T q(z_n=t)\big( logh(x_n) + E_q[\eta_t^*]^\intercal x_n - a(\eta_t^*) \big) \\

E_q[q_{\gamma_t}(V_t)] &= (\gamma_{t,1} - 1)E_q[logV_t] + (\gamma_{t,2} - 1) E_q[log(1-V_t)] - \big(log\Gamma(\gamma_{t,1} + log\Gamma(\gamma_{t,2})) -log\Gamma(\gamma_{t,1} + \gamma_{t,2})\big) \\[5pt]

E_q[q_{\tau_t}(\eta_t^*)] &= logh(\eta_t^*) + \tau_{t,1}^\intercal E_q[\eta_t^*] - \tau_{t,2} E_q[a(\eta_t^*)] - a(\tau_t) \\[5pt]

E_q[q_{\phi_n}(Z_n)] &= \phi_n^\intercal log(\phi_n) \\[10pt]

\text{where}&\\

&q(z_n=t) = \phi_{n,t} \\

&q(z_n > t) = \sum_{i=t+i}^T \phi_{n,i} \\

&E_q[logV_t] = \Psi(\gamma_{t,1}) - \Psi(\gamma_{t,1} + \gamma_{t,2}) \quad (\Psi = \text{ digamma function)} \\

&E_q[log(1-V_t)] = \Psi(\gamma_{t,2}) - \Psi(\gamma_{t,1} + \gamma_{t,2})

\\[40pt]

&\textbf{<Variational updates (CAVI)>} \\[5pt]

\gamma_{t,1} &= 1 + \sum_n \phi_{n,t} \\

\gamma_{t,2} &= \alpha + \sum_n \sum_{i=t+1}^T \phi_{n,i} \\

\tau_{t,1} &= \lambda_1 + \sum_n \phi_{n,t}x_n \\

\tau_{t,2} &= \lambda_2 + \sum_n \phi_{n,t} \\

\phi_{n,t} &\propto exp\Big(E_q[logV_t] + \sum_{i=1}^{T-1} E_q[log(1-V_i)] + E_q[\eta_t^*]^\intercal x_n - E_q[a(\eta_t^*)]\Big), \quad \Big(\sum_t \phi_{n,t} = 1\Big)

\end{aligned}
$$





