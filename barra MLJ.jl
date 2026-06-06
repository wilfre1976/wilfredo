using MLJ
using Plots
using DataFrames
using Statistics

# Configurar el motor de gráficos para VS Code
gr()

println("=" ^ 60)
println("   MÉTRICAS DE MACHINE LEARNING EN JULIA (MLJ)")
println("=" ^ 60)

# ──────────────────────────────────────────────────────────────────────────────
# PARTE 1: CLASIFICACIÓN BINARIA
# ──────────────────────────────────────────────────────────────────────────────
println("\n>>> [1/2] PROCESANDO CLASIFICACIÓN BINARIA...")

# 1. Generar datos sintéticos para clasificación
X_class, y_class = make_blobs(500, 4; centers=2, as_df=true, rng=123)
# Asegurar que las etiquetas sean tratadas como categorías (OrderedFactor)
y_class = coerce(y_class, OrderedFactor) 

# 2. Partición Entrenamiento / Test (70% - 30%)
train_idx, test_idx = partition(eachindex(y_class), 0.7, rng=42)

X_train_c, X_test_c = X_class[train_idx, :], X_class[test_idx, :]
y_train_c, y_test_c = y_class[train_idx], y_class[test_idx]

# 3. Cargar y entrenar un modelo (Logistic Regression)
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
modelo_c = LogisticClassifier()
mach_c = machine(modelo_c, X_train_c, y_train_c)
fit!(mach_c, verbose=false)

# 4. Predicciones (MLJ devuelve predicciones probabilísticas por defecto)
y_pred_proba = predict(mach_c, X_test_c)
y_pred_c = predict_mode(mach_c, X_test_c) # Predicción determinista (clases crudas)

# 5. CÁLCULO DE TODAS LAS MÉTRICAS DE CLASIFICACIÓN
println("\n--- Métricas de Clasificación ---")
@printf("Exactitud (Accuracy):         %.4f\n", accuracy(y_pred_c, y_test_c))
@printf("Precisión (Precision):       %.4f\n", precision(y_pred_c, y_test_c))
@printf("Sensibilidad (Recall/TPR):   %.4f\n", recall(y_pred_c, y_test_c))
@printf("Especificidad (Specificity):  %.4f\n", specificity(y_pred_c, y_test_c))
@printf("F1-Score:                    %.4f\n", f1score(y_pred_c, y_test_c))
@printf("Pérdida Logarítmica (LogLoss): %.4f\n", cross_entropy(y_pred_proba, y_test_c) |> mean)
@printf("Área bajo la curva ROC (AUC): %.4f\n", auc(y_pred_proba, y_test_c))

# 6. Matriz de Confusión
println("\nMatriz de Confusión:")
matriz = confusion_matrix(y_pred_c, y_test_c)
println(matriz)

# 7. Graficar Curva ROC
fprs, tprs, thresholds = roc_curve(y_pred_proba, y_test_c)
p1 = plot(fprs, tprs, linewidth=2, label="Modelo (AUC = $(round(auc(y_pred_proba, y_test_c), digits=2)))",
          title="Curva ROC - Clasificación Binaria", xlabel="FPR (1 - Especificidad)", ylabel="TPR (Sensibilidad)",
          legend=:bottomright, size=(500, 450))
plot!(p1, [0, 1], [0, 1], linestyle=:dash, color=:gray, label="Azar")


# ──────────────────────────────────────────────────────────────────────────────
# PARTE 2: REGRESIÓN
# ──────────────────────────────────────────────────────────────────────────────
println("\n\n>>> [2/2] PROCESANDO REGRESIÓN...")

# 1. Generar datos sintéticos para regresión
X_reg, y_reg = make_regression(500, 3; as_df=true, rng=456)

# 2. Partición Entrenamiento / Test
train_idx_r, test_idx_r = partition(eachindex(y_reg), 0.7, rng=789)

X_train_r, X_test_r = X_reg[train_idx_r, :], X_reg[test_idx_r, :]
y_train_r, y_test_r = y_reg[train_idx_r], y_reg[test_idx_r]

# 3. Cargar y entrenar un modelo (Linear Regressor)
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
modelo_r = LinearRegressor()
mach_r = machine(modelo_r, X_train_r, y_train_r)
fit!(mach_r, verbose=false)

# 4. Predicciones
y_pred_r = predict(mach_r, X_test_r)

# 5. CÁLCULO DE TODAS LAS MÉTRICAS DE REGRESIÓN
println("\n--- Métricas de Regresión ---")
@printf("Error Cuadrático Medio (MSE):         %.4f\n", rmse(y_pred_r, y_test_r)^2) # rmse al cuadrado
@printf("Raíz del Error Cuadrático Medio(RMSE): %.4f\n", rmse(y_pred_r, y_test_r))
@printf("Error Absoluto Medio (MAE):            %.4f\n", mae(y_pred_r, y_test_r))
@printf("Coeficiente de Determinación (R²):    %.4f\n", rsq(y_pred_r, y_test_r))

# 6. Graficar Predicciones vs Valores Reales
p2 = scatter(y_test_r, y_pred_r, alpha=0.6, label="Predicciones",
             title="Regresión: Reales vs Predichos", xlabel="Valores Reales", ylabel="Valores Predichos",
             legend=:topleft, size=(500, 450))
# Línea de identidad ideal (y = x)
lims = [minimum(y_test_r), maximum(y_test_r)]
plot!(p2, lims, lims, color=:red, linewidth=2, linestyle=:dash, label="Ideal")


# ──────────────────────────────────────────────────────────────────────────────
# DESPLIEGUE VISUAL EN VS CODE
# ──────────────────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("   PROCESO COMPLETADO")
println("=" ^ 60)
println("[!] Mostrando paneles gráficos en la pestaña 'Julia Plots' de VS Code...")

# Combinar ambos gráficos en un solo lienzo interactivo
plot_final = plot(p1, p2, layout=(1, 2), size=(1000, 450))
display(plot_final)