# 📦 Federated Learning Simulation with Flower + TensorFlow

> Simulação local de Aprendizado Federado com Flower, TensorFlow e MNIST.

Este repositório demonstra como configurar um ambiente de **Aprendizado Federado (Federated Learning)** usando o framework [**Flower (FLWR)**](https://flower.dev) com **TensorFlow**, utilizando múltiplos clientes locais simulados. Ideal para testes, ensino e validação de conceitos distribuídos.

[![Python](https://img.shields.io/badge/python-3.8--3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)](https://www.tensorflow.org/)
[![Flower](https://img.shields.io/badge/Flower-1.5.0-brightgreen)](https://flower.dev)

---

## 🧰 Requisitos

- Python **3.8** a **3.11**
- `pip` atualizado
- Ambiente virtual recomendado (`venv`)

---

## 🖥️ Instalação passo a passo (Windows/macOS)

| Etapa | macOS Terminal | Windows CMD / PowerShell |
|------|----------------|---------------------------|
| 1. Clone o repo | `git clone https://github.com/felipekumagae/federated-learning-tvbox.git`<br>`cd federated-learning-tvbox` | idem |
| 2. Crie venv | `python3 -m venv fl_env` | `python -m venv fl_env` |
| 3. Ative venv | `source fl_env/bin/activate` | `fl_env\Scripts\activate` ou `. fl_env\Scripts\Activate.ps1` |
| 4. Atualize pip | `pip install --upgrade pip` | idem |
| 5. Instale pacotes | `pip install flwr tensorflow numpy` | idem |
| 6. Rode simulação | `python fl_simulation_windows.py` | idem |
| 7. Finalize | `deactivate` | idem |

---

## 🚀 O que a simulação faz

- Inicia um servidor local (`localhost:8080`)
- Roda 3 clientes com dados diferentes do MNIST
- Cada cliente treina localmente, depois envia os pesos ao servidor
- O servidor agrega os pesos via média

---

## 📁 Estrutura do Projeto

```bash
federated-learning-tvbox/
├── fl_simulation_windows.py     # Script principal (servidor + clientes)
├── README.md                    # Este arquivo
└── fl_env/                      # Ambiente virtual (criado localmente)
```

---

## ⚙️ Personalizações possíveis

No código `fl_simulation_windows.py`, altere:

```python
num_clients = 3         # Número de clientes
num_rounds = 5          # Rounds globais
local_epochs = 1        # Épocas locais por cliente
```

---

## 🧪 Testado com

| Componente   | Versão         |
|--------------|----------------|
| Python       | 3.11.8 ✅       |
| TensorFlow   | 2.14.0 ✅       |
| Flower       | 1.5.0 ✅        |
| macOS        | Monterey 12+ ✅ |
| Windows      | 10/11 ✅        |

---

## ⚠️ Observações

- Python 3.13 ainda **não é compatível** com TensorFlow.
- Verifique se o `pip` está atualizado antes de instalar as libs.


---

## 👥 Autoria

Desenvolvido por **LINCE – Liga de Inteligência Neuro-Computacional na Engenharia**  
📍 Instituto de Ciência e Tecnologia de Sorocaba – UNESP  
🔗 https://github.com/felipekumagae/federated-learning-tvbox
