Ahora, en esta version de modelos dummys 4.0 me enfocaré en la combinación de mis tres modelos más "rentables";
    
    ssl_tl_model,
    volting_classifier_model,
    red_neuronal_mlp.

Abordaré tres enfoques, y al final voy a comparar cuál de los tres siguientes ha sido el mejor, tomé la idea de chatgpt jeje:

⚡ 1. Enfoque de Votación (Voting Ensemble)
Este método consiste en usar los tres modelos y tomar la decisión basada en el "voto" de la mayoría.

📌 Cómo funciona:

Cada modelo genera una señal:

1 = Comprar

0 = No hacer nada

-1 = Vender

Se suman las señales y se toma la decisión de mayoría:

Si 2 o más modelos dicen "comprar" → Se compra.

Si 2 o más dicen "vender" → Se vende.

Si hay empate (1 compra, 1 vende, 1 neutro) → No hacer nada.

💡 Ventajas:
✅ Reduce señales falsas de un solo modelo.
✅ Hace que las operaciones sean más estables.

⚠ Desventajas:
❌ Puede perder oportunidades si los modelos no coinciden.

📊 2. Promedio Ponderado de Confianza
Aquí no tomamos una decisión binaria (compra/venta) directamente, sino que ponderamos las señales por la confianza de cada modelo.

📌 Cómo funciona:

Cada modelo asigna una probabilidad de que el precio suba o baje (por ejemplo, un modelo puede decir "80% de probabilidad de subir").

Se ponderan estas probabilidades según su desempeño histórico (por ejemplo, si SSL-TL ha sido más confiable, se le da más peso).

Se obtiene un valor final, y si supera un umbral (ej. >60% probabilidad de subir), se compra.

💡 Ventajas:
✅ Se adapta mejor a la confiabilidad de cada modelo.
✅ Permite ajustar el sistema a lo largo del tiempo.

⚠ Desventajas:
❌ Requiere calcular bien los pesos de cada modelo.

🔄 3. Modelo Jerárquico (Cascade Model)
Aquí se usa un modelo para filtrar señales y otro para ejecutar las operaciones.

📌 Cómo funciona:

Primer filtro: Un modelo más conservador (ej. SSL-TL) decide si el mercado es propicio para operar.

Confirmación: Si el primer modelo dice que es buen momento para operar, los otros dos modelos (Volting y MLP) generan la señal de compra o venta.

Ejecución: Solo se opera cuando ambos pasos coinciden en la dirección.

💡 Ventajas:
✅ Reduce el riesgo de operar en mercados desfavorables.
✅ Evita operar con señales dudosas.

⚠ Desventajas:
❌ Puede reducir el número de operaciones, perdiendo oportunidades.
