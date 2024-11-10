import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Configuración de logging para capturar errores en un archivo log
logging.basicConfig(filename="error.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializar FastAPI
app = FastAPI()

# Obtener claves API desde las variables de entorno
llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verificar que las claves API están configuradas correctamente
if not llama_api_key or not openai_api_key:
    logging.error("Las claves de API no están configuradas correctamente en el entorno.")
    raise ValueError("Las claves de API no están configuradas.")

# Modelos de datos para las solicitudes
class DocumentRequest(BaseModel):
    file_path: str  # Ruta del archivo PDF a procesar

class QueryRequest(BaseModel):
    query: str  # Consulta para extraer datos específicos del documento procesado

# Variable global para almacenar el motor de consulta
query_engine = None

# Endpoint para procesar y leer el archivo PDF
@app.post("/process")
async def process_file(request: DocumentRequest):
    global query_engine
    try:
        # Verificar si el archivo existe en la ruta especificada
        if not os.path.isfile(request.file_path):
            logging.error(f"Archivo no encontrado en la ruta: {request.file_path}")
            raise HTTPException(status_code=400, detail="Archivo no encontrado en la ruta especificada.")
        
        # Configurar el parser
        parser = LlamaParse(result_type="text")
        
        # Leer y parsear el archivo PDF
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(input_files=[request.file_path], file_extractor=file_extractor).load_data()
        
        # Crear un índice a partir de los documentos
        index = VectorStoreIndex.from_documents(documents)

        # Crear un motor de consulta y almacenarlo globalmente
        query_engine = index.as_query_engine()

        logging.info("Archivo procesado exitosamente")
        return {"message": "Archivo procesado exitosamente"}
    except Exception as e:
        logging.error("Error en el endpoint /process: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

# Endpoint para realizar una consulta en el documento ya procesado
@app.post("/query")
async def query_document(request: QueryRequest):
    try:
        # Verificar si el motor de consulta está disponible
        if query_engine is None:
            logging.error("Intento de consulta sin procesar ningún documento")
            raise HTTPException(status_code=400, detail="No se ha procesado ningún documento aún.")

        # Instrucción completa para la consulta
        query = """
Instrucciones para extraer datos de facturas de gas y electricidad:

1. Dirección de suministro: Extrae solo el nombre de la calle o avenida sin números ni caracteres adicionales. Evita signos de puntuación o abreviaturas incorrectas.

Ejemplo:
Si la dirección es "AVD NTRA SEÑORA DE FATIMA 0036", extrae únicamente "AVD NTRA SEÑORA DE FATIMA".
Si la dirección es "C/. Francisco Gervas, N.º 9", extrae únicamente "C/. Francisco Gervas".
Nota: Si el campo está ausente, asigna "0" para indicar que no hay información.

2. Fecha de la factura: Verifica que la fecha corresponda a la emisión de la factura o "Fecha de cargo". No la confundas con las fechas de los períodos de consumo o cualquier otra fecha de referencia. Extrae la fecha en formato YYYY-MM-DD.

Ejemplo:
Si la "Fecha de cargo" es "18.12.2015", extrae "2015-12-18".
Nota: Si el campo está ausente, asigna "0".

3.Fecha desde: Corresponde a la fecha de inicio del período de facturación. Suele estar en "PERIODO DE CONSUMO", la primera fecha. Extrae esta fecha en formato YYYY-MM-DD.

Ejemplo:
Si el período de facturación comienza el "15.10.2015", extrae "2015-10-15".

4. Fecha hasta: Corresponde a la fecha de finalización del período de facturación. Suele estar en "PERIODO DE CONSUMO", la primera fecha. Extrae esta fecha en formato YYYY-MM-DD.

Ejemplo:
Si el período de facturación termina el "05.11.2015", extrae "2015-11-05".

5.Consumo en kWh (solo gas): Total de consumo de gas medido en kWh, suele aparece en la sección consumo gas. Suma ambos periodos par Este suele ser el valor total de kWh para dicho período.

Consumo en kWh (solo gas) es el kwh del impuesto especial sobre hidrocarburos.


- Busca el valor etiquetado como "Consumo gas" en la sección de gas.
- Si el consumo aparece en MWh, conviértelo a kWh multiplicando por 1000.
- Formato de extracción: 
   - Extrae solo el número seguido de "kWh", omitiendo cualquier punto o coma como separador de miles.
- Ejemplo de extracción:
   - En la factura, si el consumo aparece como "6.680 kWh", extrae "6680 kWh".
- Validación adicional:
   - Si el valor extraído es inferior a 1000 kWh, revisa visualmente para asegurarte de que no hubo error de OCR.
   - Si hay múltiples valores de consumo en períodos distintos, asegúrate de elegir el del impuesto especial sobre hidrocarburos.



6. Término fijo sin IVA: Extrae este importe de "cuota fija" o "término fijo", sumale la regulación de cuota fija si está presente para sacar el total y tener "Término fijo sin IVA"


7.Cuota variable sin IVA: Este es el importe variable suele ir asociado a "consumo gas" o a la suma de todos los importes variables unicamente de los periodos dentro del "consumo gas".

Instrucciones para extraer los importes asociados al consumo de gas en cada período, sumarlos:

Ubicación en la factura:

Dirígete a la sección "consumo gas".
Busca las líneas que describen cada período de consumo de gas y localiza el valor asociado al consumo de gas (es decir, la Cuota variable sin IVA). Este valor generalmente se encuentra alineado con el consumo en kWh y muestra un importe específico para cada período.
Qué hacer:

Suma solo los valores asociados al consumo de gas en cada período.

Ejemplo:
Consumo gas
Período de fecha xx-xx-xxxx a xx-xx-xxxx: xx.xx €
Período 2 de fecha xx-xx-xxxx a xx-xx-xxxx: yy.xx €
Cuota variable sin IVA: xx.xx + yy.xx = xx.xx €

Nota: Si el campo está ausente, asigna "0" para indicar que no hubo cuota variable.


8..Alquiler del contador sin IVA: Este es el importe del alquiler del contador en euros sin IVA.

Verifica que el importe corresponda al alquiler del contador y no a otros cargos fijos.

Ejemplo:

Si el "Alquiler del contador" es 38,57 Eur, extrae 38,57 Eur.
Nota: Si el campo está ausente, asigna "0".

9.Impuesto especial sobre hidrocarburos sin IVA: Este impuesto específico suele aparecer en la sección de gas y se aplica a un período específico de consumo de gas.

Ejemplo:
Si el "Impuesto especial sobre hidrocarburos" es 93,50 Eur, extrae 93,50 Eur.

Nota: Si el campo no se menciona en la factura, asigna "0" ya que es un campo opcional.

10.Descuento en gas: 

Para el descuento del gas, en caso de que aparezca un % de descuento. 
Cálculo del Descuento con porcentaje  :

Extrae  porcentaje de descuento aplicable al valor base de consumo.
Si el descuento está basado en un porcentaje, realiza el cálculo usando el valor base multiplicado por el % indicado.

Procedimiento de Cálculo:

Ejemplo:
Si el valor base es 1901,77 € y el % de descuento es 3%, sería 0,03 realiza el cálculo:
1901.77*0.03=57.05€
1901.77*0.03=57.05€
Extrae este valor calculado para "descuento_gas".

Nota: Si el campo no está presente, asigna "0" sin hacer suposiciones



11. Total electricidad: Este campo debe extraerse solo si existe un apartado específico para electricidad en la factura, identificado como "total electricidad" o un subtotal de la sección de electricidad.

Ejemplo:
Si el "Total electricidad" es 392,99 Eur, extrae 392,99 Eur.
Nota: Si el campo no está presente, asigna "0" sin hacer suposiciones.

Posibles errores y cómo evitarlos:

Confusión de fechas: Distingue claramente entre la "Fecha de cargo" (para la "Fecha de la factura") y las fechas del período de consumo (para "Fecha desde" y "Fecha hasta").

Unidades de consumo (kWh vs. MWh): Comprueba siempre la unidad en la que está medido el consumo. Si el consumo está en MWh, conviértelo a kWh multiplicando por 1000.

Diferenciación de cuotas fijas y variables: Verifica cada campo en su sección correspondiente para evitar confundir el "término fijo" con la cuota variable de consumo o los descuentos con cuotas variables.

Impuestos específicos: Algunos impuestos, como el impuesto especial sobre hidrocarburos, pueden confundirse con otros impuestos. Asegúrate de leer la descripción del impuesto para identificarlo correctamente.

Presencia de la sección de electricidad: Solo extrae "Total electricidad" si se especifica claramente en la factura; si no está presente, asigna "0" sin hacer suposiciones.

Múltiples períodos de consumo de gas (Cuota variable): Si existen varios períodos de consumo de gas, suma los importes variables de cada período, asegurándote de que todos pertenecen a la cuota variable y excluyendo descuentos y otros cargos.
Formato JSON esperado

Asegúrate de estructurar los datos extraídos en el siguiente formato JSON:

json
Copiar código
{
   "direccion_suministro": "Nombre de la calle o avenida sin números",
   "fecha_factura": "YYYY-MM-DD",
   "fecha_desde": "YYYY-MM-DD",
   "fecha_hasta": "YYYY-MM-DD",
   "consumo_kWh": "XX,XX kWh",
   "termino_fijo_sin_iva": "XX,XX Eur",
   "cuota_variable_sin_iva": "XX,XX Eur",
   "alquiler_contador_sin_iva": "XX,XX Eur",
   "impuesto_esp_hidrocarburo_sin_iva": "XX,XX Eur",
   "descuento_gas": "XX,XX Eur",
   "total_electricidad": "XX,XX Eur"
}

Nota: Reemplaza "[INSERTAR AQUÍ EL TEXTO REAL DE LA FACTURA]" con el texto de la factura que deseas analizar.
"""
        
        # Realizar la consulta en el documento
        try:
            response = query_engine.query(query)
            logging.info("Consulta realizada exitosamente")
        except Exception as e:
            logging.error("Error en query_engine.query: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Error en la consulta del motor: {str(e)}")

        return {"response": response}
    except Exception as e:
        logging.error("Error en el endpoint /query: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error al realizar la consulta: {str(e)}")

        raise HTTPException(status_code=500, detail=f"Error al realizar la consulta: {str(e)}")
