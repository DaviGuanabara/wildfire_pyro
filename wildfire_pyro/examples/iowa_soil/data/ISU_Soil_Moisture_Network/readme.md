# 📄 **README — Iowa Environmental Mesonet (IEM) Dataset**

## 🌎 **Fonte de dados**

Este conjunto de dados foi coletado a partir do **Iowa Environmental Mesonet (IEM)**, administrado pela Iowa State University.  

- **Estações meteorológicas (ISUSM Network):**  
  [IEM Stations — ISUSM Network](https://mesonet.agron.iastate.edu/sites/networks.php?network=ISUSM)

- **Dados climáticos diários:**  
  [AgClimate Daily Data Tool](https://mesonet.agron.iastate.edu/agclimate/hist/daily.php)

---

## 🛰️ **Estações selecionadas**

As estações foram selecionadas manualmente na interface oficial do IEM. Informações detalhadas (como posição exata no mapa, coordenadas e metadados) podem ser obtidas diretamente em:  
[AgClimate Daily — Station Metadata](https://mesonet.agron.iastate.edu/agclimate/hist/daily.php)

Estações principais (exemplo):  
- Ames — Horticulture (Vineyard)
- Bankston — Park Farm Winery (Vineyard)
- Oskaloosa — Tassel Ridge (Vineyard)
- Glenwood — Blackwing Vineyard (Vineyard)
- Ames-AEA (SoilVue)
- Ames-Kitchen (SoilVue)
- Jefferson (SoilVue)

---

## 🌤️ **Variáveis selecionadas**

Na interface de download (`https://mesonet.agron.iastate.edu/agclimate/hist/daily.php`), foram **marcadas as seguintes caixas**:

### **Variáveis gerais (climáticas):**

- High Temperature [F]
- Low Temperature [F]
- Minimum Relative Humidity [%]
- Average Relative Humidity [%]
- Maximum Relative Humidity [%]
- Solar Radiation [MJ/m²]
- Precipitation [inch]
- Average Wind Speed [mph]
- Wind Gust [mph]
- Reference Evapotranspiration [inch]
- Barometric Pressure Average [mb]

---

### **Variáveis específicas de vinhedos (Vineyard Station-only Variables):**

- lwmv_1
- lwmv_2
- lwmdry_1_tot
- lwmdry_2_tot
- lwmcon_1_tot
- lwmcon_2_tot
- lwmwet_1_tot
- lwmwet_2_tot

---

### **Variáveis de solo (SoilVue — nas estações específicas):**

- Soil temperature and moisture at multiple depths:
  - sv_t2, sv_vwc2
  - sv_t4, sv_vwc4
  - sv_t8, sv_vwc8
  - sv_t12, sv_vwc12
  - sv_t14, sv_vwc14
  - sv_t16, sv_vwc16
  - sv_t20, sv_vwc20
  - sv_t24, sv_vwc24
  - sv_t28, sv_vwc28
  - sv_t30, sv_vwc30
  - sv_t32, sv_vwc32
  - sv_t36, sv_vwc36
  - sv_t40, sv_vwc40
  - sv_t42, sv_vwc42
  - sv_t52, sv_vwc52

---

## 💾 **Formato e flags**

- **Formato de download:** Microsoft Excel (.xlsx)
- **Missing values:** Representados como células vazias
- **Quality Control Flags:** Incluídas (colunas terminadas em `_f`), posteriormente removidas no processamento final
- **Download direto para o disco:** Sim (caixa marcada)

---

## ⚙️ **Processamento realizado**

- Merge com metadados das estações (incluindo coordenadas)
- Renomeação e padronização das colunas
- Remoção de colunas vazias e flags de qualidade
- Remoção de colunas de metadados desnecessárias
- Exclusão de linhas com valores faltantes
- Export final para arquivo `dataset_cleaned.xlsx`

---

## 🗺️ **Posição geográfica**

As coordenadas (latitude e longitude), elevação e demais atributos das estações podem ser consultados em:  
[ISUSM Station Metadata](https://mesonet.agron.iastate.edu/sites/networks.php?network=ISUSM)

---

## ✉️ **Contato e observações**

Este dataset foi coletado para fins de pesquisa em microclima e previsão de condições para viticultura (molhamento foliar, risco de doenças, etc.).  
Em caso de dúvidas ou sugestões, entrar em contato com o responsável técnico do projeto.

---

## ✅ **Status**

- 🔎 Dados baixados e revisados
- 💻 Dataset final limpo salvo
- 📄 Documentação consolidada (este README)
