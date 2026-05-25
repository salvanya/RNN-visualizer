# Visualizador interactivo de RNNs (GRU / LSTM)

App web estática, en español, pensada para una cátedra universitaria de Deep Learning. Permite recorrer paso a paso el flujo de información en redes recurrentes (**GRU** y **LSTM**) sobre dos arquitecturas:

- **Análisis de sentimiento**: encoder de 2 capas + cabeza densa + softmax sobre 3 clases (negativo / neutro / positivo).
- **Traducción seq2seq**: encoder–decoder, con y sin **mecanismo de atención** (tipo Luong general).

El objetivo es hacer visualmente explícitos los puntos que más cuesta transmitir en pizarrón: el desdoblamiento temporal de una misma celda, el flujo del vector `h_{t-1}` completo hacia cada unidad, las dimensiones de cada vector y matriz, la diferencia entre encoder (input fijo) y decoder (autoregresivo), y cómo cambia el flujo al activar atención.

**Demo en vivo**: <https://salvanya.github.io/RNN-visualizer/>

## Cómo se usa

- Cambiá entre **GRU** y **LSTM** desde la barra superior.
- Elegí modo **sentimiento** o **traducción**; en traducción podés activar **atención**.
- Avanzá con la barra de timesteps (click directo, drag para scrubbing) o con play/pause/stop.
- Atajos: `←` / `→` para avanzar timestep, `Espacio` para play/pause, `Esc` para cerrar modales y limpiar tooltips.
- **Hover** sobre cualquier vector muestra sus valores; **click** fija el tooltip. Se pueden mantener varios tooltips abiertos simultáneamente.
- **Click sobre una celda** abre el modal interno con las gates animadas y sus fórmulas.

Todo el ejemplo está precomputado: la frase de entrada (`"Me encantó la pizza de este lugar"`) y la traducción (`"I loved this place's pizza"`) son fijas. La app no entrena ni calcula — solo navega un JSON con los estados de cada paso.

## Desarrollo

Requisitos: **Node 20+** y, para regenerar datos, **Python 3.10+** con `numpy`.

```bash
npm install
npm run dev      # servidor de desarrollo en http://localhost:5173
npm run build    # build de producción a /dist
npm run preview  # sirve /dist localmente
```

## Regenerar el JSON precomputado

El frontend lee `src/data/data.json`, que se genera offline desde Python:

```bash
pip install -r scripts/requirements.txt
python scripts/generate_data.py
```

El script ejecuta el forward pass real con pesos hardcodeados para los 6 escenarios (GRU/LSTM × sentimiento / traducción sin atención / traducción con atención) y verifica la coherencia numérica de cada paso con tolerancia `1e-6`.

## Despliegue

- **GitHub Pages**: cada push a `main` dispara `.github/workflows/deploy.yml`, que buildea y publica a Pages.
- `vite.config.ts` usa `base: '/RNN-visualizer/'` para que los assets resuelvan bajo esa ruta.

## Stack

Vite · React 19 · TypeScript · Tailwind v4 · Framer Motion · Zustand · KaTeX · lucide-react.

Más detalles de diseño, dimensiones, decisiones cerradas y roadmap están en [`CLAUDE.md`](./CLAUDE.md).
