// Default `tools` (OpenAI function-calling schema) the Playground sends when
// tool calling is enabled. Editable in the UI; this is just the starting set.

export const DEFAULT_TOOLS = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather in a given location",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string", description: "The city and state, e.g. Kuala Lumpur, MY" },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] },
        },
        required: ["location"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_stock_price",
      description: "Get the latest stock price for a given ticker symbol",
      parameters: {
        type: "object",
        properties: {
          ticker: { type: "string", description: "Stock ticker symbol, e.g. AAPL, MAYBANK" },
          exchange: { type: "string", description: "Exchange code, e.g. NASDAQ, KLSE" },
        },
        required: ["ticker"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_flights",
      description: "Search for available flights between two airports on a given date",
      parameters: {
        type: "object",
        properties: {
          origin: { type: "string", description: "IATA code of departure airport, e.g. KUL" },
          destination: { type: "string", description: "IATA code of arrival airport, e.g. SIN" },
          date: { type: "string", description: "Departure date in YYYY-MM-DD format" },
          passengers: { type: "integer", description: "Number of passengers", default: 1 },
        },
        required: ["origin", "destination", "date"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "send_email",
      description: "Send an email to a recipient",
      parameters: {
        type: "object",
        properties: {
          to: { type: "string", description: "Recipient email address" },
          subject: { type: "string", description: "Email subject line" },
          body: { type: "string", description: "Email body content" },
          cc: { type: "array", items: { type: "string" }, description: "Optional list of CC recipients" },
        },
        required: ["to", "subject", "body"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculate",
      description: "Evaluate a math expression and return the result",
      parameters: {
        type: "object",
        properties: {
          expression: { type: "string", description: "A math expression, e.g. '2 * (3 + 4)'" },
        },
        required: ["expression"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "translate_text",
      description: "Translate text from one language to another",
      parameters: {
        type: "object",
        properties: {
          text: { type: "string", description: "The text to translate" },
          source_lang: { type: "string", description: "Source language code, e.g. 'en', 'ms', 'zh'" },
          target_lang: { type: "string", description: "Target language code, e.g. 'en', 'ms', 'zh'" },
        },
        required: ["text", "target_lang"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "create_calendar_event",
      description: "Create an event on the user's calendar",
      parameters: {
        type: "object",
        properties: {
          title: { type: "string", description: "Event title" },
          start_time: { type: "string", description: "ISO 8601 start datetime, e.g. 2026-05-10T14:00:00+08:00" },
          end_time: { type: "string", description: "ISO 8601 end datetime" },
          attendees: { type: "array", items: { type: "string" }, description: "List of attendee emails" },
          location: { type: "string", description: "Optional location" },
        },
        required: ["title", "start_time", "end_time"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web and return top results",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          top_k: { type: "integer", description: "Number of results to return", default: 5 },
        },
        required: ["query"],
      },
    },
  },
] as const;

export const DEFAULT_TOOLS_JSON = JSON.stringify(DEFAULT_TOOLS, null, 2);
