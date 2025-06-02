import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Body, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERVICE_API_KEY_EXPECTED = os.getenv("SERVICE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY не найден в .env файле или переменных окружения.")
if not SERVICE_API_KEY_EXPECTED:
    raise ValueError("SERVICE_API_KEY не найден в .env файле или переменных окружения. Авторизация не будет работать.")

genai.configure(api_key=GEMINI_API_KEY)

class GenerateSchemaRequest(BaseModel):
    user_prompt: str

class NodePosition(BaseModel):
    x: float
    y: float

class NodeButton(BaseModel):
    id: str = Field(..., examples=["btn_1699898200123_abc12"])
    text: str = Field(..., examples=["Далее"])
    callback_data: str = Field(..., examples=["cb_next_step"])

class NodeMappingRule(BaseModel):
    path: str = Field(..., examples=["data.user.name"])
    variableName: str = Field(..., examples=["userName"])

class NodeSqlParameter(BaseModel):
    variableName: str = Field(..., examples=["userId"])
    dataType: str = Field(default="string", examples=["string", "integer", "float", "boolean"])

class NodeOutputMapping(BaseModel):
    column: str = Field(..., examples=["user_email_column"])
    variable: str = Field(..., examples=["userEmailVar"])

class BaseNodeData(BaseModel):
    label: str = Field(..., examples=["Мой узел"])

class StartNodeData(BaseNodeData):
    command: str = Field(default="/start", examples=["/register"])

class MessageNodeData(BaseNodeData):
    messageText: str = Field(..., examples=["Привет, {userName}!"])
    buttons: List[NodeButton] = Field(default_factory=list)

class ConditionNodeData(BaseNodeData):
    conditionType: str = Field(default="variable_check", examples=["variable_check", "user_reply"])
    variableName: Optional[str] = Field(default=None, examples=["userAge"])
    operator: Optional[str] = Field(default="equals_text", examples=["equals_text", "is_number_greater_than"])
    valueToCompare: Optional[Any] = Field(default=None, examples=["active", 18])
    replyOperator: Optional[str] = Field(default=None, examples=["equals_text", "contains_text"])
    replyValue: Optional[str] = Field(default=None, examples=["cb_yes"])

class UserInputNodeData(BaseNodeData):
    questionText: str = Field(..., examples=["Как вас зовут?"])
    variableToStore: str = Field(..., examples=["userName"])

class ApiCallNodeData(BaseNodeData):
    url: str = Field(..., examples=["https://api.example.com/data"])
    method: str = Field(default="GET", examples=["GET", "POST"])
    headers: str = Field(default="{}", examples=['{"Content-Type": "application/json"}']) 
    body: str = Field(default="{}", examples=['{"key": "value"}']) 
    variableToStoreSuccess: str = Field(default="api_raw_response", examples=["apiData"])
    variableToStoreError: str = Field(default="api_error", examples=["apiErrorLog"])

class ExtractDataNodeData(BaseNodeData):
    inputVariable: str = Field(..., examples=["api_raw_response"])
    mappings: List[NodeMappingRule] = Field(default_factory=list)

class SetVariableNodeData(BaseNodeData):
    variableName: str = Field(..., examples=["myVariable"])
    variableValue: Any = Field(..., examples=["some_text", 123, '{"json_key": "json_value"}'])

class StorageNodeData(BaseNodeData):
    storageDefinitionSlug: Optional[str] = Field(default=None, examples=["user_profiles_storage"])
    operation: str = Field(default="set_value", examples=["set_value", "get_value", "delete_value", "check_key", "increment_value", "decrement_value"])
    scope: str = Field(default="scope_user", examples=["scope_user", "scope_bot"])
    storageKey: str = Field(..., examples=["user_points_{_flow_user_id_}"])
    valueToSet: Optional[Any] = Field(default=None, examples=["initial_value", 100])
    isJsonString: Optional[bool] = Field(default=False)
    resultVariableName: Optional[str] = Field(default=None, examples=["retrievedValue", "keyExists", "updatedCounter"])
    stepValue: Optional[float] = Field(default=1.0)

class DatabaseNodeData(BaseNodeData):
    selectedIntegrationId: Optional[str] = Field(default=None, examples=["db_integration_uuid"]) 
    integrationName: Optional[str] = Field(default=None, examples=["Основная Пользовательская БД"]) 
    queryType: str = Field(default="select_single", examples=["select_single", "select_multiple", "execute_dml"])
    queryTypeLabel: Optional[str] = Field(default=None, examples=["SELECT (одна строка)"]) 
    sqlQuery: str = Field(..., examples=["SELECT * FROM users WHERE id = ?;"])
    parameters: List[NodeSqlParameter] = Field(default_factory=list)
    outputMappings: List[NodeOutputMapping] = Field(default_factory=list) 
    resultListVariable: Optional[str] = Field(default="db_results_list", examples=["userList"]) 
    affectedRowsVariable: Optional[str] = Field(default="affected_rows_count", examples=["updatedUserCount"]) 
    analyzedTableName: Optional[str] = Field(default=None) 

class NodeModel(BaseModel):
    id: str = Field(..., examples=["node_unique_id_1"])
    type: str = Field(..., examples=["messageNode"])
    position: NodePosition
    data: Dict[str, Any]

class EdgeModel(BaseModel):
    id: str = Field(..., examples=["edge_1_to_2"])
    source: str = Field(..., examples=["node_unique_id_1"])
    target: str = Field(..., examples=["node_unique_id_2"])
    sourceHandle: Optional[str] = Field(default=None, examples=["start_output", "true_output"])
    targetHandle: Optional[str] = Field(default=None, examples=["message_input", "condition_input"])
    type: str = Field(default="smoothstep")

class ReactFlowSchema(BaseModel):
    nodes: List[NodeModel]
    edges: List[EdgeModel]
    viewport: Optional[Dict[str, Any]] = Field(default=None)

app = FastAPI(
    title="Nodera AI Schema Generator",
    description="Сервис для генерации React Flow схем для Nodera с помощью Gemini AI.",
    version="0.1.1"
)

API_KEY_NAME = "X-API-Key"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header_auth)):
    """
    Проверяет предоставленный API ключ.
    """
    if api_key_header == SERVICE_API_KEY_EXPECTED:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, detail="Неверный или отсутствующий API ключ"
        )

SYSTEM_PROMPT_TEMPLATE = """
Ты — продвинутый ИИ-ассистент, специализирующийся на создании структур (схем) для конструктора Telegram-ботов Nodera.
Твоя задача — по текстовому запросу пользователя сгенерировать JSON-структуру, представляющую собой граф узлов и связей для платформы React Flow.
Эта структура будет использоваться для автоматического создания схемы бота. Nodera позволяет визуально конструировать логику Telegram-ботов.
Пользователи могут создавать несколько независимых потоков логики, каждый из которых инициируется своей командой (например, /start, /help, /shop).

ВАЖНО: Твой ответ ДОЛЖЕН БЫТЬ ТОЛЬКО JSON объектом и НИЧЕМ БОЛЕЕ. Не добавляй никакого описательного текста до или после JSON.
JSON должен соответствовать следующей структуре:
{{{{
  "nodes": [
    {{{{
      "id": "node_УНИКАЛЬНЫЙ_ID",
      "type": "ТИП_УЗЛА",
      "position": {{ "x": X_КООРДИНАТА, "y": Y_КООРДИНАТА }},
      "data": {{
        "label": "Метка узла (на русском)",
        // ... другие поля data, специфичные для ТИПА_УЗЛА ...
      }}
    }}}}
  ],
  "edges": [
    {{{{
      "id": "edge_УНИКАЛЬНЫЙ_ID",
      "source": "ID_ИСХОДНОГО_УЗЛА",
      "target": "ID_ЦЕЛЕВОГО_УЗЛА",
      "sourceHandle": "ID_ИСХОДНОГО_ПОРТА", 
      "targetHandle": "ID_ЦЕЛЕВОГО_ПОРТА", 
      "type": "smoothstep"
    }}}}
  ]
}}}}

Доступные типы узлов (ТИП_УЗЛА) и их основные поля в `data` (всегда включай поле "label" на русском языке):
{node_type_descriptions}

Рекомендации по генерации схемы:
1.  **Множественные команды/потоки:** Если запрос пользователя подразумевает несколько независимых сценариев, инициируемых разными командами (например, "/start" для приветствия и "/feedback" для сбора отзывов), создай для каждой такой команды свой узел "startNode". Располагай начальные узлы разных потоков на достаточном расстоянии друг от друга по горизонтали (например, с шагом X в 500-600 единиц), чтобы потоки визуально не пересекались. Первый "startNode" может быть на x: 250, y: 50. Второй "startNode" на x: 800, y: 50, и так далее.
2.  **Расположение узлов внутри одного потока:** Узлы одного потока располагай преимущественно вертикально вниз. Стандартное расстояние между узлами по оси Y ~150-200 единиц. Для ветвлений (после "conditionNode") располагай узлы веток "Да" и "Нет" по бокам от основной оси потока, например, сдвигая их по X на +/- 150-200 единиц относительно узла условия, и на том же или следующем уровне по Y.
3.  **ID узлов и связей:** Используй осмысленные и уникальные `id` (например, "node_start_main", "node_ask_name", "edge_start_to_ask_name"). Если генерируешь несколько потоков, ID узлов и ребер не должны пересекаться между потоками.
4.  **Метки узлов (`label`):** Должны быть краткими, понятными и на русском языке.
5.  **Обработка кнопок ("messageNode"):**
    - Если подразумеваются кнопки, добавь в `data` массив `buttons`. Каждый объект кнопки: `{{"id": "btn_УНИК_ID", "text": "Текст кнопки", "callback_data": "уник_callback_data"}}`.
    - Исходящее ребро от такой кнопки должно использовать `sourceHandle` вида `btn-out-ID_КНОПКИ` (например, `btn-out-btn_action1`).
    - Если у "messageNode" нет кнопок и есть простой следующий шаг, используй `sourceHandle: "message_output_main"`.
6.  **Узел "conditionNode":**
    - `conditionType`: "variable_check" или "user_reply".
    - Для "variable_check": `variableName`, `operator`, `valueToCompare`. `valueToCompare` должно быть строкой, даже если это число (например, "18").
    - Для "user_reply": `replyOperator`, `replyValue` (ожидаемое callback_data).
    - Всегда два исходящих ребра: одно с `sourceHandle: "true_output"`, другое с `sourceHandle: "false_output"`.
7.  **Узел "userInputNode":** `questionText` и `variableToStore` (имя переменной для ответа).
8.  **Переменные:** Имена переменных в `variableToStore` (userInputNode), `variableName` (conditionNode, setVariableNode), `inputVariable` (extractDataNode), `resultVariableName` (storageNode) и т.д. должны быть в camelCase или snake_case и не должны содержать фигурных скобок `{{}}`. Платформа Nodera сама будет обрабатывать их как плейсхолдеры вида `{{{{имя_переменной}}}}` при использовании в текстах сообщений или других полях.
9.  **Узлы API, Хранилища, БД:** Заполняй ключевые поля (`url`, `method` для API; `operation`, `storageKey` для Хранилища; `sqlQuery`, `queryType` для БД) и необходимые переменные для результатов. `headers` и `body` в `apiCallNode` должны быть строками, содержащими валидный JSON.
10. **Логика по умолчанию:** Если запрос пользователя неясный, создай простую схему с "startNode" и "messageNode" с приветствием.
11. **Только JSON:** Никакого текста до или после JSON-ответа. Убедись, что JSON строго соответствует описанной структуре.

Текущий запрос пользователя: "{user_prompt}"
Сгенерируй JSON-структуру для этого запроса.
"""

NODE_TYPE_DESCRIPTIONS = f"""
- "startNode": Стартовый узел бота. Инициирует поток логики по команде.
  - data: {{{{ "label": "Начало", "command": "/start" }}}}
    - `command`: (string) Текстовая команда для запуска этого потока (например, "/shop", "/register"). Должна начинаться с "/".
  - Выходные порты (sourceHandle): `start_output` (основной выход).

- "messageNode": Узел для отправки текстового сообщения пользователю. Может содержать inline-кнопки.
  - data: {{{{ "label": "Сообщение", "messageText": "Текст сообщения...", "buttons": [{{{{ "id": "btn_1", "text": "Кнопка 1", "callback_data": "cb_1" }}}}] }}}}
    - `messageText`: (string) Текст сообщения. Можно использовать плейсхолдеры (например, `{{{{userName}}}}`).
    - `buttons`: (array) Массив объектов кнопок (опционально). Каждая кнопка:
        - `id`: (string) Уникальный ID кнопки (например, "btn_confirm_order"). Важно для `sourceHandle`.
        - `text`: (string) Текст на кнопке.
        - `callback_data`: (string) Строка, отправляемая при нажатии.
  - Входные порты (targetHandle): `message_input` (основной вход).
  - Выходные порты (sourceHandle):
    - `message_output_main`: Если нет кнопок и есть следующий шаг.
    - `btn-out-ID_КНОПКИ`: Для каждой кнопки, где ID_КНОПКИ - это `id` из объекта кнопки (например, `btn-out-btn_confirm_order`).

- "conditionNode": Узел для ветвления логики на основе условия.
  - data: {{{{ "label": "Условие", "conditionType": "variable_check", "variableName": "varName", "operator": "equals_text", "valueToCompare": "someValue", "replyOperator": "equals_text", "replyValue": "cb_data" }}}}
    - `conditionType`: (string) Тип условия: "variable_check" или "user_reply".
    - `variableName`: (string, optional) Имя переменной для проверки (для `conditionType: "variable_check"`).
    - `operator`: (string, optional) Оператор сравнения для переменной (например, "equals_text", "contains_text", "is_number_greater_than", "is_number_less_than", "is_number_equal_to").
    - `valueToCompare`: (string|number, optional) Значение для сравнения с переменной. Для числовых сравнений передавай как строку (например, "18").
    - `replyOperator`: (string, optional) Оператор сравнения для `callback_data` (для `conditionType: "user_reply"`, например, "equals_text", "starts_with_text").
    - `replyValue`: (string, optional) Ожидаемое значение `callback_data`.
  - Входные порты (targetHandle): `condition_input`.
  - Выходные порты (sourceHandle): `true_output` (если условие истинно), `false_output` (если условие ложно).

- "userInputNode": Запрашивает текстовый ввод у пользователя.
  - data: {{{{ "label": "Ввод пользователя", "questionText": "Введите ваше имя:", "variableToStore": "userName" }}}}
    - `questionText`: (string) Текст вопроса, который увидит пользователь.
    - `variableToStore`: (string) Имя переменной, в которую будет сохранен ответ пользователя.
  - Входные порты (targetHandle): `input_A`.
  - Выходные порты (sourceHandle): `output_A`.

- "apiCallNode": Выполняет HTTP-запрос к внешнему API.
  - data: {{{{ "label": "API запрос", "url": "https://...", "method": "GET", "headers": "{{{{}}}}", "body": "{{{{}}}}", "variableToStoreSuccess": "api_response", "variableToStoreError": "api_error" }}}}
    - `url`: (string) URL API-эндпоинта.
    - `method`: (string) HTTP-метод ("GET", "POST", "PUT", "DELETE", "PATCH").
    - `headers`: (string) JSON-строка с заголовками (например, `{{"Authorization": "Bearer TOKEN"}}`). По умолчанию пустой JSON-объект "{{{{}}}}".
    - `body`: (string) JSON-строка с телом запроса (для POST, PUT, PATCH). По умолчанию пустой JSON-объект "{{{{}}}}".
    - `variableToStoreSuccess`: (string) Имя переменной для сохранения успешного ответа.
    - `variableToStoreError`: (string) Имя переменной для сохранения информации об ошибке.
  - Входные порты (targetHandle): `api_input`.
  - Выходные порты (sourceHandle): `api_output_success`, `api_output_error`.

- "extractDataNode": Извлекает данные из JSON-объекта (хранящегося в переменной) с помощью JSONPath.
  - data: {{{{ "label": "Извлечь JSON", "inputVariable": "api_response", "mappings": [{{{{ "path": "user.name", "variableName": "extractedUserName" }}}}] }}}}
    - `inputVariable`: (string) Имя переменной, содержащей JSON.
    - `mappings`: (array) Массив правил извлечения. Каждое правило:
        - `path`: (string) JSONPath-выражение (например, `data.items[0].price`).
        - `variableName`: (string) Имя новой переменной для сохранения извлеченного значения.
  - Входные порты (targetHandle): `extract_input`.
  - Выходные порты (sourceHandle): `extract_output`.

- "setVariableNode": Устанавливает или обновляет значение переменной в потоке.
  - data: {{{{ "label": "Установить переменную", "variableName": "myVar", "variableValue": "некое значение" }}}}
    - `variableName`: (string) Имя переменной.
    - `variableValue`: (any) Значение переменной (может быть строкой, числом, JSON-строкой).
  - Входные порты (targetHandle): `setvar_input`.
  - Выходные порты (sourceHandle): `setvar_output`.

- "storageNode": Взаимодействие с внутренним хранилищем данных бота.
  - data: {{{{ "label": "Хранилище", "operation": "set_value", "scope": "scope_user", "storageKey": "user_data", "valueToSet": "some data", "resultVariableName": "stored_data", "isJsonString": false, "stepValue": 1, "storageDefinitionSlug": null }}}}
    - `operation`: (string) Тип операции ("set_value", "get_value", "delete_value", "check_key", "increment_value", "decrement_value").
    - `scope`: (string) Область видимости ключа ("scope_user", "scope_bot").
    - `storageKey`: (string) Ключ для данных.
    - `valueToSet`: (any, optional) Значение для записи (для "set_value").
    - `isJsonString`: (boolean, optional) Если true, `valueToSet` будет сохранено как JSON-объект/массив.
    - `resultVariableName`: (string, optional) Имя переменной для сохранения результата.
    - `stepValue`: (number, optional) Числовой шаг для "increment_value", "decrement_value".
    - `storageDefinitionSlug`: (string, optional) Слаг определения хранилища.
  - Входные порты (targetHandle): `storage_input`.
  - Выходные порты (sourceHandle): `storage_output_next`.

- "databaseNode": Выполнение SQL-запросов к внешним базам данных.
  - data: {{{{ "label": "Запрос БД", "selectedIntegrationId": null, "queryType": "select_single", "sqlQuery": "SELECT * FROM table WHERE id = ?", "parameters": [{{{{ "variableName": "_flow_user_id_", "dataType": "integer" }}}}], "outputMappings": [{{{{ "column": "email", "variable": "userEmail" }}}}], "resultListVariable": "results", "affectedRowsVariable": "count" }}}}
    - `selectedIntegrationId`: (string|null) ID существующей интеграции с БД.
    - `queryType`: (string) Тип запроса ("select_single", "select_multiple", "execute_dml").
    - `sqlQuery`: (string) Текст SQL-запроса. Используй `?` для плейсхолдеров параметров.
    - `parameters`: (array) Массив объектов параметров. Каждый объект: `{{{{ "variableName": "имя_переменной", "dataType": "тип_данных" }}}}`.
    - `outputMappings`: (array, optional, для "select_single") Массив правил маппинга колонок. Каждый объект: `{{{{ "column": "имя_колонки", "variable": "имя_переменной" }}}}`.
    - `resultListVariable`: (string, optional, для "select_multiple") Имя переменной для списка строк.
    - `affectedRowsVariable`: (string, optional, для "execute_dml") Имя переменной для кол-ва измененных строк.
  - Входные порты (targetHandle): `db_input`.
  - Выходные порты (sourceHandle): `db_output_success`, `db_output_error`.
"""

ALLOWED_NODE_TYPES = [
    "startNode", "messageNode", "conditionNode", "userInputNode",
    "apiCallNode", "extractDataNode", "setVariableNode", "storageNode", "databaseNode"
]

def validate_generated_schema(schema_json: dict) -> tuple[Optional[ReactFlowSchema], Optional[str]]:
    """
    Валидирует JSON-схему, сгенерированную AI.
    """
    try:
        schema = ReactFlowSchema(**schema_json)
        if not schema.nodes:
            return None, "Сгенерированная схема не содержит узлов (nodes)."
        node_ids = set()
        for node_idx, node in enumerate(schema.nodes):
            if not node.id:
                 node.id = f"node_gen_{node_idx + 1}" 
            if node.id in node_ids:
                original_id = node.id
                counter = 1
                while f"{original_id}_{counter}" in node_ids:
                    counter += 1
                node.id = f"{original_id}_{counter}"
            node_ids.add(node.id)
            if node.type not in ALLOWED_NODE_TYPES:
                return None, f"Недопустимый тип узла: {node.type} для узла {node.id}"
            if not node.data.get("label"):
                 node.data["label"] = f"{node.type.replace('Node','').capitalize()} {node.id.split('_')[-1]}"
            required_data_fields = {
                "startNode": ["command"], "messageNode": ["messageText"], "conditionNode": ["conditionType"],
                "userInputNode": ["questionText", "variableToStore"],
                "apiCallNode": ["url", "method", "variableToStoreSuccess", "variableToStoreError"],
                "extractDataNode": ["inputVariable", "mappings"],
                "setVariableNode": ["variableName", "variableValue"],
                "storageNode": ["operation", "storageKey"], "databaseNode": ["sqlQuery", "queryType"],
            }
            if node.type in required_data_fields:
                for field in required_data_fields[node.type]:
                    if field not in node.data:
                         pass 
        edge_ids = set()
        for edge_idx, edge in enumerate(schema.edges):
            if not edge.id:
                edge.id = f"edge_gen_{edge_idx + 1}"
            if edge.id in edge_ids:
                original_id = edge.id
                counter = 1
                while f"{original_id}_{counter}" in edge_ids:
                    counter += 1
                edge.id = f"{original_id}_{counter}"
            edge_ids.add(edge.id)
            if edge.source not in node_ids:
                return None, f"Связь {edge.id} ссылается на несуществующий исходный узел {edge.source}"
            if edge.target not in node_ids:
                return None, f"Связь {edge.id} ссылается на несуществующий целевой узел {edge.target}"
        return schema, None
    except Exception as e:
        return None, f"Ошибка валидации Pydantic или внутренняя ошибка валидации схемы: {str(e)}"

@app.post("/generate-schema", response_model=ReactFlowSchema, response_model_exclude_none=True)
async def generate_schema_endpoint(
    request_data: GenerateSchemaRequest = Body(...),
    api_key: str = Depends(get_api_key)
):
    """
    Принимает текстовый запрос пользователя и генерирует схему React Flow.
    Требует валидный X-API-Key в заголовках.
    """
    user_prompt_text = request_data.user_prompt
    if not user_prompt_text.strip():
        raise HTTPException(status_code=400, detail="Запрос пользователя не может быть пустым.")
    full_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt_text,
        node_type_descriptions=NODE_TYPE_DESCRIPTIONS
    )
    generation_config = genai.types.GenerationConfig(
        temperature=0.2, 
        top_p=0.9,
        top_k=30,
    )
    safety_settings = [ 
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            system_instruction=full_system_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = model.generate_content(user_prompt_text)
        generated_text = response.text.strip()
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason_message = f"Запрос был заблокирован AI. Причина: {response.prompt_feedback.block_reason}."
            if response.prompt_feedback.safety_ratings:
                 block_reason_message += f" Рейтинги безопасности: {response.prompt_feedback.safety_ratings}"
            raise HTTPException(status_code=400, detail=block_reason_message)
        if generated_text.startswith("```json"):
            generated_text = generated_text[len("```json"):].strip()
        elif generated_text.startswith("```"):
            generated_text = generated_text[len("```"):].strip()
        if generated_text.endswith("```"):
            generated_text = generated_text[:-len("```")].strip()
        if not generated_text:
            raise HTTPException(status_code=500, detail="AI не вернул никакого текста.")
        try:
            schema_json = json.loads(generated_text)
        except json.JSONDecodeError as e:
            error_detail_msg = f"AI вернул невалидный JSON. Ошибка: {e}. Получено: '{generated_text[:500]}...'"
            print(f"Ошибка декодирования JSON от AI: {e}")
            print(f"Полученный текст: \n---\n{generated_text}\n---")
            raise HTTPException(status_code=500, detail=error_detail_msg)
        validated_schema, error_detail = validate_generated_schema(schema_json)
        if error_detail:
            print(f"Ошибка валидации сгенерированной схемы: {error_detail}")
            print(f"Схема JSON: \n---\n{json.dumps(schema_json, indent=2, ensure_ascii=False)}\n---")
            raise HTTPException(status_code=500, detail=f"Сгенерированная схема не прошла валидацию: {error_detail}")
        if not validated_schema:
            raise HTTPException(status_code=500, detail="Не удалось провалидировать схему после генерации (неизвестная ошибка).")
        return validated_schema
    except Exception as e:
        print(f"Произошла ошибка при вызове Gemini API или обработке ответа: {type(e).__name__}: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера при генерации схемы AI: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)