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

## 🖥️ Instalação passo a passo
## 🖥️ Instalação passo a passo (Windows/macOS)

> 💡 Recomendado: use Python 3.10 para garantir compatibilidade com TensorFlow

| Etapa | macOS Terminal | Windows CMD / PowerShell |
|------|----------------|---------------------------|
| 1. Clone o repo | `git clone https://github.com/felipekumagae/federated-learning-tvbox.git`<br>`cd federated-learning-tvbox` | idem |
| 2. Apague qualquer ambiente antigo | `rm -rf fl_env` | `rmdir /S /Q fl_env` |
| 3. Crie venv com Python 3.10 | `python3.10 -m venv fl_env` | `python -m venv fl_env` (se estiver usando Python 3.10) |
| 4. Ative o venv | `source fl_env/bin/activate` | `fl_env\Scripts\activate` ou `.\fl_env\Scripts\Activate.ps1` |
| 5. Atualize pip | `pip install --upgrade pip` | idem |
| 6. Instale pacotes | `pip install tensorflow flwr numpy` | idem |
| 7. Teste o TensorFlow | `python -c "import tensorflow as tf; print(tf.__version__)"` | idem |
| 8. Rode simulação | `python fl_simu.py` | idem |
| 9. Finalize | `deactivate` | idem |

------|----------------|---------------------------|
| 1. Clone o repo | `git clone https://github.com/felipekumagae/federated-learning-tvbox.git`<br>`cd federated-learning-tvbox` | idem |
| 2. Crie venv | `python3 -m venv fl_env` | `python -m venv fl_env` |
| 3. Ative venv | `source fl_env/bin/activate` | `fl_env\Scripts\activate` ou `. fl_env\Scripts\Activate.ps1` |
| 4. Atualize pip | `pip install --upgrade pip` | idem |
| 5. Instale pacotes | `pip install flwr tensorflow numpy` | idem |
| 6. Rode simulação | `python fl_simu.py` | idem |
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
├── fl_simu.py     # Script principal (servidor + clientes)
├── README.md                    # Este arquivo
└── fl_env/                      # Ambiente virtual (criado localmente)
```

---

## ⚙️ Personalizações possíveis

No código `fl_simu.py`, altere:

```python
num_clients = 3         # Número de clientes
num_rounds = 5          # Rounds globais
local_epochs = 1        # Épocas locais por cliente
```

---

## 🧪 Testado com

| Componente   | Versão         |
|--------------|----------------|
| Python       | 3.10 ✅       |
| TensorFlow   | 2.19.0 ✅       |
| Flower       | 1.17.0 ✅        |
| macOS        | Monterey 12+ ✅ |
| Windows      | 10/11 ✅        |

---

## ⚠️ Observações

- Python 3.13 ainda **não é compatível** com TensorFlow.
- Verifique se o `pip` está atualizado antes de instalar as libs.


---

## 👥 Autoria

Desenvolvido por **Felipe Kumagae - LINCE (Liga de Inteligência Neuro-Computacional na Engenharia)**  
📍 Instituto de Ciência e Tecnologia de Sorocaba – UNESP  
🔗 https://github.com/felipekumagae/federated-learning-tvbox
