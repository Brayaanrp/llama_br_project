# Cargar las variables de entorno
from dotenv import load_dotenv
import os
load_dotenv()

# Importar dependencias necesarias
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Configurar el parser
parser = LlamaParse(result_type="text")  # Puedes cambiar "markdown" a "text" si prefieres solo texto.

# Leer y parsear el archivo PDF
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['47 2018-11.pdf'], file_extractor=file_extractor).load_data()

# Crear un índice a partir de los documentos
index = VectorStoreIndex.from_documents(documents)

# Crear un motor de consulta
query_engine = index.as_query_engine()

# Realizar una consulta de ejemplo
query = """
Extrae la siguiente información clave de este texto de factura:

1.Fecha de la factura: Fecha en la que se emitió la factura, en formato YYYY-MM-DD.
2.Fecha desde: Fecha de inicio del período de facturación, en formato YYYY-MM-DD.
3.Fecha hasta: Fecha de finalización del período de facturación, en formato YYYY-MM-DD.
4.Consumo en kWh: Total de consumo medido en kWh. Nota: Si el consumo aparece en MWh, conviértelo a kWh (1 MWh = 1000 kWh).
5. Cuota fija sin IVA: Importe fijo en euros sin IVA.
6.Cuota variable sin IVA (Consumo gas): Importe variable asociado al consumo de gas, en euros sin IVA.
7.Alquiler del contador sin IVA: Importe del alquiler del contador en euros sin IVA. Nota: Si el campo está ausente, asigna 0 para indicar que no hubo alquiler de contador.
8.Impuesto especial sobre hidrocarburos sin IVA: Valor del impuesto sobre hidrocarburos en euros sin IVA. Nota: Si no se menciona este campo, asigna 0 ya que es un campo opcional.
9.Descuento en gas: Descuento aplicado al consumo de gas, en euros sin IVA. Nota: Si el campo está ausente, asigna 0 para indicar que no hubo descuento.
10Total electricidad: Solo debe extraerse si existe un apartado específico para electricidad en la factura, identificado como "total electricidad". Si este apartado no está presente, asigna 0.

Aquí está el texto de la factura: [INSERTAR AQUÍ EL TEXTO REAL DE LA FACTURA]

Extrae los datos en el siguiente formato JSON:

{
   "fecha_factura": "YYYY-MM-DD",
   "fecha_desde": "YYYY-MM-DD",
   "fecha_hasta": "YYYY-MM-DD",
   "consumo_kWh": "XX.XX kWh",
   "cuota_fija_sin_iva": "XX.XX Eur",
   "cuota_variable_sin_iva": "XX.XX Eur",
   "alquiler_contador_sin_iva": "XX.XX Eur",
   "impuesto_esp_hidrocarburo_sin_iva": "XX.XX Eur",
   "descuento_gas": "XX.XX Eur",
   "total_electricidad": "XX.XX Eur"
}

Nota: Reemplaza "[INSERTAR AQUÍ EL TEXTO REAL DE LA FACTURA]" con el texto de la factura que deseas analizar.
"""
response = query_engine.query(query)
print(response)  # Imprime la respuesta de la consulta
