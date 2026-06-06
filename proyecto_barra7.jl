using Plots
using LaTeXStrings
using Printf
using LinearAlgebra
import Pkg; Pkg.add(["Plots", "LaTeXStrings"])
using Statistics # Necesario para la función mean()

# ── 1. PARÁMETROS DEL PROBLEMA ─────────────────────────────────────────────────
const L  = 0.80       # Longitud de la barra [m]
const k  = 1.17e-4    # Difusividad térmica del cobre [m²/s]
const T1 = 10.0       # Temperatura extremo izquierdo [°C]  (T1 ∈ [4, 15])
const T2 = 60.0       # Temperatura extremo derecho   [°C]  (T2 ∈ [30, 80])
const Ti = 23.0       # Temperatura inicial uniforme  [°C]  (≈ temperatura ambiente)
const N  = 100        # Número de términos de la serie de Fourier

# Mallas
const Nx    = 200     # Puntos espaciales
const Nt    = 600     # Puntos temporales
const t_max = 300.0   # Duración total [s] = 5 minutos

x = LinRange(0.0, L, Nx)
t = LinRange(0.0, t_max, Nt)

println("=" ^ 60)
println("  ECUACIÓN DEL CALOR 1D — Solución Analítica")
println("=" ^ 60)
@printf("  L  = %.2f m\n",  L)
@printf("  k  = %.2e m²/s\n", k)
@printf("  T1 = %.1f °C\n", T1)
@printf("  T2 = %.1f °C\n", T2)
@printf("  Ti = %.1f °C\n", Ti)
@printf("  N  = %d términos\n", N)
println("=" ^ 60)

# ── 2. SOLUCIÓN ESTACIONARIA ───────────────────────────────────────────────────
"""Distribución de temperatura en estado estacionario (lineal)."""
u_e(x_val) = T1 + (T2 - T1) / L * x_val

# ── 3. COEFICIENTES DE FOURIER ─────────────────────────────────────────────────
"""
Calcula el n-ésimo coeficiente de Fourier mediante integración numérica
(regla del trapecio con 2000 puntos).

    Bₙ = (2/L) ∫₀ᴸ [Ti − u_e(ξ)] · sin(nπξ/L) dξ
"""
function fourier_coeff(n::Int)
    xi   = LinRange(0.0, L, 2000)
    λn   = n * π / L
    f    = @. (Ti - u_e(xi)) * sin(λn * xi)
    return (2.0 / L) * sum((f[1:end-1] .+ f[2:end]) ./ 2 .* step(xi))
end

println("\nCalculando coeficientes de Fourier...")
Bn = [fourier_coeff(n) for n in 1:N]
println("  Listo. Primeros 3 coeficientes: B1=$(round(Bn[1],digits=4)),",
        " B2=$(round(Bn[2],digits=4)), B3=$(round(Bn[3],digits=4))")

# ── 4. SOLUCIÓN COMPLETA u(x, t) ──────────────────────────────────────────────
"""
Evalúa la solución analítica u(x, t) con N términos de la serie de Fourier.
"""
function u_analytic(x_val::Float64, t_val::Float64)
    resultado = u_e(x_val)
    for n in 1:N
        λn = n * π / L
        resultado += Bn[n] * sin(λn * x_val) * exp(-k * λn^2 * t_val)
    end
    return resultado
end

# Construir superficie completa U[i, j] = u(x[i], t[j])
println("\nCalculando superficie de temperatura u(x, t)...")
U = [u_analytic(xi, tj) for xi in x, tj in t]
println("  Listo. Dimensiones de U: $(size(U))")

# ── Verificación ──────────────────────────────────────────────────────────────
@printf("\nVerificación de condiciones:\n")
@printf("  u(0,   300s) = %.4f °C  (esperado T1 = %.1f °C)\n", U[1,   end], T1)
@printf("  u(L,   300s) = %.4f °C  (esperado T2 = %.1f °C)\n", U[end, end], T2)
@printf("  u(L/2,   0s) = %.4f °C  (esperado Ti = %.1f °C)\n",
        u_analytic(L/2, 0.0), Ti)
@printf("  u(L/2, 300s) = %.4f °C  (estado est. = %.1f °C)\n",
        u_analytic(L/2, 300.0), u_e(L/2))

# ── 5. FIGURA 1: SUPERFICIE 3D ────────────────────────────────────────────────
println("\nGenerando Figura 1: superficie 3D...")

x_cm = collect(x) .* 100   # convertir a cm para el eje

p1 = surface(
    x_cm, collect(t), U',
    xlabel = "Posición x [cm]",
    ylabel = "Tiempo t [s]",
    zlabel = "Temperatura [°C]",
    title  = "Distribución de temperatura u(x, t)\nBarra de cobre — Ecuación del calor 1D",
    color  = :plasma,
    alpha  = 0.92,
    camera = (55, 30),
    size   = (900, 600),
    colorbar_title = " T [°C]",
)
savefig(p1, "fig1_superficie_3D.png")
println("  Guardada: fig1_superficie_3D.png")

# ── 6. FIGURA 2: EVOLUCIÓN TEMPORAL POR SENSOR ────────────────────────────────
println("Generando Figura 2: curvas temporales por sensor...")

sensores_m = 0.0:0.10:L          # posiciones en metros: 0, 10, 20, ..., 80 cm
n_s        = length(sensores_m)
palette_s  = cgrad(:RdYlBu, n_s, rev=true)

p2 = plot(
    title  = "Evolución temporal en cada sensor\n(líneas punteadas = estado estacionario)",
    xlabel = "Tiempo [s]",
    ylabel = "Temperatura [°C]",
    legend = :outerright,
    size   = (1000, 550),
    grid   = true,
    gridalpha = 0.3,
)

for (idx, xs) in enumerate(sensores_m)
    T_sensor = [u_analytic(xs, tj) for tj in t]
    col = palette_s[idx]
    label_str = @sprintf("x = %d cm", round(Int, xs * 100))
    plot!(p2, collect(t), T_sensor, label=label_str, color=col, lw=2.0)
    hline!(p2, [u_e(xs)], color=col, lw=0.8, ls=:dash, label=false)
end

savefig(p2, "fig2_sensores_temporales.png")
println("  Guardada: fig2_sensores_temporales.png")

# ── 7. FIGURA 3: PERFILES ESPACIALES A DISTINTOS TIEMPOS ─────────────────────
println("Generando Figura 3: perfiles espaciales...")

tiempos_plot = [0.0, 5.0, 15.0, 30.0, 60.0, 120.0, 180.0, 300.0]
palette_t    = cgrad(:viridis, length(tiempos_plot))

p3 = plot(
    title  = "Perfiles de temperatura en distintos instantes\nBarra de cobre (L = 80 cm)",
    xlabel = "Posición x [cm]",
    ylabel = "Temperatura [°C]",
    legend = :outerright,
    size   = (900, 550),
    grid   = true,
    gridalpha = 0.3,
)

for (idx, tp) in enumerate(tiempos_plot)
    perfil = [u_analytic(xi, tp) for xi in x]
    col    = palette_t[idx]
    label_str = @sprintf("t = %.0f s", tp)
    plot!(p3, x_cm, perfil, label=label_str, color=col, lw=2.0)
end

plot!(p3, x_cm, u_e.(x), label="Estado estacionario", color=:black,
      lw=2.5, ls=:dash)

savefig(p3, "fig3_perfiles_espaciales.png")
println("  Guardada: fig3_perfiles_espaciales.png")

# ── 8. FIGURA 4: CONVERGENCIA DE LA SERIE ─────────────────────────────────────
println("Generando Figura 4: análisis de convergencia...")

t_conv     = 30.0   # instante de análisis
N_tests    = [1, 5, 10, 20, 50, 100]
u_ref      = [u_analytic(xi, t_conv) for xi in x]   # solución con N=100

# 4a — Perfiles con distintos N
p4a = plot(
    title  = @sprintf("Convergencia de la serie — t = %.0f s", t_conv),
    xlabel = "Posición x [cm]",
    ylabel = "Temperatura [°C]",
    legend = :outerright,
    size   = (600, 450),
    grid   = true, gridalpha = 0.3,
)

pal4 = palette(:Set1, length(N_tests))
for (idx, Nt_) in enumerate(N_tests)
    perfil = u_e.(x)
    for n in 1:Nt_
        λn = n * π / L
        perfil .+= Bn[n] .* sin.(λn .* x) .* exp(-k * λn^2 * t_conv)
    end
    plot!(p4a, x_cm, perfil, label="N = $Nt_", color=pal4[idx], lw=1.8)
end

# 4b — Error MAE vs N
errores = Float64[]
for Nt_ in N_tests
    perfil = u_e.(x)
    for n in 1:Nt_
        λn = n * π / L
        perfil .+= Bn[n] .* sin.(λn .* x) .* exp(-k * λn^2 * t_conv)
    end
    push!(errores, mean(abs.(perfil .- u_ref)))
end

p4b = plot(
    N_tests, errores,
    title  = "Error MAE vs número de términos N",
    xlabel = "Número de términos N",
    ylabel = "MAE [°C] (vs N=100)",
    legend = :topright,
    size   = (600, 450),
    marker = :circle, markersize = 7,
    color  = "#185FA5", lw = 2.0,
    yscale = :log10,
    grid   = true, gridalpha = 0.3, minorgrid = true,
    label  = "MAE",
)
hline!(p4b, [0.01], color=:red, ls=:dash, lw=1.5, label="Umbral 0.01 °C")

p4 = plot(p4a, p4b, layout=(1, 2), size=(1100, 480))
savefig(p4, "fig4_convergencia.png")
println("  Guardada: fig4_convergencia.png")

# ── 9. RESUMEN FINAL ──────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("  RESUMEN DE RESULTADOS")
println("=" ^ 60)
@printf("  Tiempo característico τ₁ = 1/(k·(π/L)²) = %.1f s\n",
        1 / (k * (π / L)^2))
@printf("  Temperatura estacionaria en x = L/2: %.2f °C\n", u_e(L/2))

println("\n  Análisis de convergencia (t = 30 s):")
for (Nt_, err) in zip(N_tests, errores)
    @printf("    N = %3d → MAE = %.6f °C\n", Nt_, err)
end

println("\n  Figuras generadas:")
println("    fig1_superficie_3D.png")
println("    fig2_sensores_temporales.png")
println("    fig3_perfiles_espaciales.png")
println("    fig4_convergencia.png")
println("=" ^ 60)
println("\n✓ Simulación completada exitosamente.")