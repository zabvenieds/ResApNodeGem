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
    id: str = Field(..., examples=["btn_action_1"])
    text: str = Field(..., examples=["Подтвердить"])
    callback_data: str = Field(..., examples=["confirm_action_1"])

class NodeMappingRule(BaseModel):
    path: str = Field(..., examples=["data.user.profile.name"])
    variableName: str = Field(..., examples=["userNameFromProfile"])

class NodeSqlParameter(BaseModel):
    variableName: str = Field(..., examples=["targetUserId"])
    dataType: str = Field(default="string", examples=["string", "integer", "float", "boolean"])

class NodeOutputMapping(BaseModel):
    column: str = Field(..., examples=["email_address"])
    variable: str = Field(..., examples=["userEmailVariable"])

class BaseNodeData(BaseModel):
    label: str = Field(..., examples=["Основной узел"])

class StartNodeData(BaseNodeData):
    command: str = Field(default="/start", examples=["/begin_process"])

class MessageNodeData(BaseNodeData):
    messageText: str = Field(..., examples=["Добро пожаловать, {{{{userName}}}}! Как я могу помочь?"])
    buttons: List[NodeButton] = Field(default_factory=list)

class ConditionNodeData(BaseNodeData):
    conditionType: str = Field(default="variable_check", examples=["variable_check", "user_reply"])
    variableName: Optional[str] = Field(default=None, examples=["userAgeValue"])
    operator: Optional[str] = Field(default="equals_text", examples=["equals_text", "is_number_greater_than", "contains_text"])
    valueToCompare: Optional[Any] = Field(default=None, examples=["completed_status", "18", "substring_to_find"])
    replyOperator: Optional[str] = Field(default=None, examples=["equals_text", "starts_with_text"])
    replyValue: Optional[str] = Field(default=None, examples=["callback_yes_option"])

class UserInputNodeData(BaseNodeData):
    questionText: str = Field(..., examples=["Пожалуйста, введите ваше полное имя:"])
    variableToStore: str = Field(..., examples=["fullUserName"])

class ApiCallNodeData(BaseNodeData):
    url: str = Field(..., examples=["https://api.service.com/v1/data_resource"])
    method: str = Field(default="GET", examples=["GET", "POST", "PUT", "DELETE"])
    headers: str = Field(default="{{}}", examples=['{{"Authorization": "Bearer YOUR_TOKEN", "X-Custom-Header": "Value"}}']) 
    body: str = Field(default="{{}}", examples=['{{"item_id": 123, "quantity": 2}}']) 
    variableToStoreSuccess: str = Field(default="api_service_response", examples=["serviceData"])
    variableToStoreError: str = Field(default="api_service_error", examples=["serviceErrorLog"])

class ExtractDataNodeData(BaseNodeData):
    inputVariable: str = Field(..., examples=["api_service_response"])
    mappings: List[NodeMappingRule] = Field(default_factory=list)

class SetVariableNodeData(BaseNodeData):
    variableName: str = Field(..., examples=["processed_user_status"])
    variableValue: Any = Field(..., examples=["active_user", 100, '{{"order_id": 55, "status": "paid"}}'])

class StorageNodeData(BaseNodeData):
    storageDefinitionSlug: Optional[str] = Field(default=None, examples=["main_user_data_storage"])
    operation: str = Field(default="set_value", examples=["set_value", "get_value", "delete_value", "check_key", "increment_value", "decrement_value"])
    scope: str = Field(default="scope_user", examples=["scope_user", "scope_bot"])
    storageKey: str = Field(..., examples=["user_profile_{{{{_flow_user_id_}}}}"])
    valueToSet: Optional[Any] = Field(default=None, examples=["some_string_data", 50, '{{"key":"complex_value"}}'])
    isJsonString: Optional[bool] = Field(default=False)
    resultVariableName: Optional[str] = Field(default=None, examples=["retrieved_storage_value", "key_existence_flag", "updated_counter_value"])
    stepValue: Optional[float] = Field(default=1.0)

class DatabaseNodeData(BaseNodeData):
    selectedIntegrationId: Optional[str] = Field(default=None, examples=["pgsql_production_db_id"]) 
    integrationName: Optional[str] = Field(default=None, examples=["Основная База Данных Проекта"]) 
    queryType: str = Field(default="select_single", examples=["select_single", "select_multiple", "execute_dml"])
    queryTypeLabel: Optional[str] = Field(default=None, examples=["SELECT (одна строка из таблицы)"]) 
    sqlQuery: str = Field(..., examples=["SELECT user_name, registration_date FROM users WHERE user_id = ? AND status = ?;"])
    parameters: List[NodeSqlParameter] = Field(default_factory=list)
    outputMappings: List[NodeOutputMapping] = Field(default_factory=list) 
    resultListVariable: Optional[str] = Field(default="database_query_results_list", examples=["activeUserList"]) 
    affectedRowsVariable: Optional[str] = Field(default="database_affected_rows", examples=["deletedUserCount"]) 
    analyzedTableName: Optional[str] = Field(default=None) 

class NodeModel(BaseModel):
    id: str = Field(..., examples=["node_start_flow_1"])
    type: str = Field(..., examples=["messageNode"])
    position: NodePosition
    data: Dict[str, Any]

class EdgeModel(BaseModel):
    id: str = Field(..., examples=["edge_node1_to_node2"])
    source: str = Field(..., examples=["node_start_flow_1"])
    target: str = Field(..., examples=["node_initial_message"])
    sourceHandle: Optional[str] = Field(default=None, examples=["start_output", "true_output_condition1"])
    targetHandle: Optional[str] = Field(default=None, examples=["message_input_main", "condition_input_main"])
    type: str = Field(default="smoothstep")

class ReactFlowSchema(BaseModel):
    nodes: List[NodeModel]
    edges: List[EdgeModel]
    viewport: Optional[Dict[str, Any]] = Field(default=None)

app = FastAPI(
    title="Nodera AI Schema Generator",
    description="Сервис для генерации React Flow схем для Nodera с помощью Gemini AI. Nodera - это no-code конструктор Telegram ботов.",
    version="0.1.3"
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
Ты — ИИ-ассистент для платформы Nodera. Nodera — это визуальный no-code конструктор Telegram-ботов, где пользователи строят логику бота, соединяя различные функциональные узлы на графе React Flow. Каждый узел выполняет определенное действие (отправка сообщения, запрос ввода, условие, вызов API, работа с данными и т.д.). Твоя задача — по текстовому описанию от пользователя сгенерировать JSON-структуру этой схемы. JSON должен быть валидным и строго соответствовать формату React Flow, используемому в Nodera.

ВАЖНО: Ответ должен содержать ТОЛЬКО JSON-объект. Никакого сопроводительного текста, объяснений или markdown-форматирования вокруг JSON.

Структура JSON:
{{{{
  "nodes": [
    {{{{
      "id": "уникальный_строковый_id_узла_латиницей_без_пробелов", 
      "type": "ТИП_УЗЛА_ИЗ_СПИСКА_НИЖЕ",
      "position": {{{{ "x": число, "y": число }}}}, 
      "data": {{{{ 
        "label": "Понятная метка узла на русском языке", 
        // ... другие поля data ...
      }}}}
    }}}}
  ],
  "edges": [
    {{{{
      "id": "уникальный_строковый_id_связи_латиницей_без_пробелов", 
      "source": "id_исходного_узла",
      "target": "id_целевого_узла",
      "sourceHandle": "id_выходного_порта_на_source_узле_опционально", 
      "targetHandle": "id_входного_порта_на_target_узле_опционально", 
      "type": "smoothstep" 
    }}}}
  ]
}}}}

Описание доступных типов узлов (ТИП_УЗЛА) и их полей в `data`:
{node_type_descriptions}

Ключевые правила и рекомендации по генерации:
1.  **Имена переменных:** При определении имен переменных (например, в `variableToStore`, `variableName`, `resultVariableName`) **НЕ ИСПОЛЬЗУЙ** фигурные скобки `{{}}`. Имена должны быть простыми строками, например: `userName`, `order_status`. Платформа Nodera будет использовать эти имена для создания плейсхолдеров вида `{{{{имя_переменной}}}}` при их использовании в текстовых полях.
2.  **Множественные команды/потоки:** Если запрос подразумевает несколько независимых сценариев, запускаемых разными командами (например, "/start", "/help"), создай для каждой команды свой узел "startNode". Располагай начальные узлы разных потоков на значительном расстоянии по горизонтали (шаг X примерно 600-800 единиц). Первый "startNode" обычно на (x: 250, y: 50).
3.  **Расположение узлов в потоке:** Узлы внутри одного логического потока располагай преимущественно вертикально вниз, с шагом по Y примерно 150-200 единиц. Для ветвлений (после "conditionNode") узлы веток "Да" и "Нет" располагай по бокам от основной оси (сдвиг по X на +/- 200-250 единиц от узла условия) на том же или следующем уровне по Y.
4.  **ID узлов и связей:** Должны быть уникальными, строковыми, на латинице, без пробелов (можно использовать подчеркивания или дефисы). Старайся делать их осмысленными.
5.  **Метки узлов (`label`):** Всегда включай это поле. Метка должна быть краткой, понятной, на русском языке.
6.  **Узел "messageNode":** `messageText` может содержать плейсхолдеры `{{{{имя_переменной}}}}`. Кнопки в массиве `buttons`, каждая: `{{"id": "btn_id", "text": "Текст", "callback_data": "cb_data"}}`. Связь от кнопки: `sourceHandle: "btn-out-ID_КНОПКИ"`. Без кнопок: `sourceHandle: "message_output_main"`.
7.  **Узел "conditionNode":** `conditionType`: "variable_check" или "user_reply". Для "variable_check": `variableName` (без `{{}}`), `operator`, `valueToCompare` (строка, даже если число, напр., "18"; может содержать плейсхолдеры). Для "user_reply": `replyOperator`, `replyValue`. Выходы: `sourceHandle: "true_output"` и `sourceHandle: "false_output"`.
8.  **Плейсхолдеры:** Используй `{{{{имя_переменной}}}}` (включая `{{{{_flow_user_id_}}}}`) для подстановки значений в поля `messageText`, `url`, `headers` (внутри JSON-строки), `body` (внутри JSON-строки), `valueToSet`, `storageKey`, `sqlQuery` и т.д.
9.  **JSON в строках:** Для `headers` и `body` узла "apiCallNode" значения должны быть строками, содержащими валидный JSON. Например, `headers: "{{\\"Content-Type\\":\\"application/json\\"}}"`.
10. **Детализация и умолчания:** Чем подробнее запрос, тем лучше. Если неясно, генерируй базовую логику.
11. **Строго JSON:** Ответ – исключительно JSON-объект.

Текущий запрос пользователя: "{user_prompt}"
Сгенерируй JSON-структуру для этого запроса.
"""

NODE_TYPE_DESCRIPTIONS = f"""
- "startNode": Стартовый узел. Инициирует поток по Telegram-команде.
  - data: {{{{ "label": "Начало (метка)", "command": "/my_command" }}}}
    - `command`: (string) Команда запуска (напр., "/start"). Обязательно с "/".
  - Выходные порты (sourceHandle): `start_output`.

- "messageNode": Отправка сообщения пользователю. Может иметь inline-кнопки.
  - data: {{{{ "label": "Сообщение", "messageText": "Текст, можно с {{{{переменной}}}}.", "buttons": [{{{{ "id": "button1_id", "text": "Кнопка", "callback_data": "action1" }}}}] }}}}
    - `messageText`: (string) Текст. Поддерживает плейсхолдеры `{{{{имя_переменной}}}}`.
    - `buttons`: (array, optional) Кнопки: `{{{{ "id": "str_id", "text": "str_text", "callback_data": "str_cb" }}}}`. ID уникален в узле.
  - Входные порты (targetHandle): `message_input`.
  - Выходные порты (sourceHandle): `message_output_main` (без кнопок), или `btn-out-ID_КНОПКИ`.

- "conditionNode": Ветвление логики.
  - data: {{{{ "label": "Условие", "conditionType": "variable_check", "variableName": "userScore", "operator": "is_number_greater_than", "valueToCompare": "100" }}}}
    - `conditionType`: (string) "variable_check" или "user_reply".
    - `variableName`: (string, optional для "variable_check") Имя переменной (без `{{}}`).
    - `operator`: (string, optional для "variable_check") Оператор (напр., "equals_text", "is_number_greater_than").
    - `valueToCompare`: (string, optional для "variable_check") Значение для сравнения (числа строкой, напр., "100"). Можно плейсхолдеры.
    - `replyOperator`: (string, optional для "user_reply") Оператор для callback_data (напр., "equals_text").
    - `replyValue`: (string, optional для "user_reply") Ожидаемое callback_data.
  - Входные порты (targetHandle): `condition_input`.
  - Выходные порты (sourceHandle): `true_output`, `false_output`.

- "userInputNode": Запрос текстового ввода.
  - data: {{{{ "label": "Запрос ввода", "questionText": "Ваше имя?", "variableToStore": "inputUserName" }}}}
    - `questionText`: (string) Текст вопроса. Можно плейсхолдеры.
    - `variableToStore`: (string) Имя переменной для ответа (без `{{}}`).
  - Входные порты (targetHandle): `input_A`.
  - Выходные порты (sourceHandle): `output_A`.

- "apiCallNode": Вызов внешнего API.
  - data: {{{{ "label": "Вызов API", "url": "https://api.example.com/data?id={{{{userId}}}}", "method": "GET", "headers": "{{{{}}}}", "body": "{{{{}}}}", "variableToStoreSuccess": "apiData", "variableToStoreError": "apiError" }}}}
    - `url`: (string) URL. Можно плейсхолдеры.
    - `method`: (string) "GET", "POST", и т.д.
    - `headers`: (string) JSON-строка заголовков. Напр., `{{"Authorization": "Bearer {{{{apiToken}}}}"}}`. По умолч. `"{{{{}}}}"`.
    - `body`: (string) JSON-строка тела. Напр., `{{"name": "{{{{userName}}}}"}}`. По умолч. `"{{{{}}}}"`.
    - `variableToStoreSuccess`: (string) Имя переменной для успешного ответа (без `{{}}`).
    - `variableToStoreError`: (string) Имя переменной для ошибки (без `{{}}`).
  - Входные порты (targetHandle): `api_input`.
  - Выходные порты (sourceHandle): `api_output_success`, `api_output_error`.

- "extractDataNode": Извлечение данных из JSON (в переменной) по JSONPath.
  - data: {{{{ "label": "Извлечь JSON", "inputVariable": "jsonSourceVar", "mappings": [{{{{ "path": "user.profile.name", "variableName": "extractedName" }}}}] }}}}
    - `inputVariable`: (string) Имя переменной с JSON (без `{{}}`).
    - `mappings`: (array) Правила: `{{{{ "path": "JSONPath_выражение", "variableName": "имя_новой_переменной_без_{{}}" }}}}`.
  - Входные порты (targetHandle): `extract_input`.
  - Выходные порты (sourceHandle): `extract_output`.

- "setVariableNode": Установка/обновление значения переменной.
  - data: {{{{ "label": "Установить Переменную", "variableName": "statusVar", "variableValue": "активно" }}}}
    - `variableName`: (string) Имя переменной (без `{{}}`).
    - `variableValue`: (any) Значение (строка, число, boolean, JSON-строка). Можно плейсхолдеры.
  - Входные порты (targetHandle): `setvar_input`.
  - Выходные порты (sourceHandle): `setvar_output`.

- "storageNode": Работа с внутренним хранилищем Nodera.
  - data: {{{{ "label": "Хранилище", "operation": "set_value", "scope": "scope_user", "storageKey": "user_pref_{{{{_flow_user_id_}}}}", "valueToSet": "{{{{userChoices}}}}", "isJsonString": true }}}}
    - `operation`: (string) "set_value", "get_value", "delete_value", "check_key", "increment_value", "decrement_value".
    - `scope`: (string) "scope_user" или "scope_bot".
    - `storageKey`: (string) Ключ. Можно плейсхолдеры.
    - `valueToSet`: (any, opt) Значение для "set_value". Можно плейсхолдеры.
    - `isJsonString`: (boolean, opt) Если `true`, `valueToSet` сохраняется как JSON.
    - `resultVariableName`: (string, opt) Имя переменной для результата (get, check, increment, decrement) (без `{{}}`).
    - `stepValue`: (number, opt) Шаг для increment/decrement. По умолч. 1.
    - `storageDefinitionSlug`: (string, opt) Слаг определения хранилища.
  - Входные порты (targetHandle): `storage_input`.
  - Выходные порты (sourceHandle): `storage_output_next`.

- "databaseNode": SQL-запросы к внешним БД.
  - data: {{{{ "label": "Запрос к БД", "queryType": "select_single", "sqlQuery": "SELECT ...", "parameters": [], "outputMappings": [] }}}}
    - `selectedIntegrationId`: (string|null) ID интеграции с БД.
    - `queryType`: (string) "select_single", "select_multiple", "execute_dml".
    - `sqlQuery`: (string) SQL-запрос с `?` для параметров.
    - `parameters`: (array, opt) Параметры: `{{{{ "variableName": "имя_Nodera_переменной_без_{{}}", "dataType": "sql_тип" }}}}`.
    - `outputMappings`: (array, opt, для "select_single") Маппинг колонок: `{{{{ "column": "имя_БД_колонки", "variable": "имя_Nodera_переменной_без_{{}}" }}}}`.
    - `resultListVariable`: (string, opt, для "select_multiple") Имя переменной для списка результатов (без `{{}}`).
    - `affectedRowsVariable`: (string, opt, для "execute_dml") Имя переменной для кол-ва измененных строк (без `{{}}`).
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
                    if field not in node.data and node.data.get(field) is None : 
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
        temperature=0.15, 
        top_p=0.9,
        top_k=20,
    )
    safety_settings = [ 
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
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