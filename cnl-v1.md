# AI Interaction Language (AIL) Specification

## Core Principles
1. Every command must specify WHAT (action), ON (target), WITH (parameters), and FOR (purpose/intent)
2. All parameters must be explicitly typed
3. Each statement must have exactly one primary verb
4. No implicit references - all objects must be fully qualified
5. No ambiguous quantifiers or modifiers

## Statement Structure
```
COMMAND: <verb> ON <target> WITH <parameters> FOR <purpose>
```

## Parameter Declaration
```
PARAM <name>: <type> = <value>
CONTEXT <name>: <type> = <value>
```

## Examples

### Code Generation
```
COMMAND: generate
ON: function
WITH:
  PARAM name: string = "calculateDistance"
  PARAM inputs: list[string] = ["point1", "point2"]
  PARAM returnType: string = "float"
  PARAM language: string = "python"
  CONTEXT purpose: string = "compute Euclidean distance between 2D points"
  CONTEXT constraints: list[string] = ["handle invalid inputs", "optimize for readability"]
FOR: implementation

// Instead of ambiguous "write a function that calculates distance"
```

### Data Analysis
```
COMMAND: analyze
ON: dataset
WITH:
  PARAM source: string = "sales_data.csv"
  PARAM metrics: list[string] = ["mean", "median", "trend"]
  PARAM timeframe: daterange = "2023-01-01 TO 2024-01-01"
  CONTEXT significance: float = 0.05
  CONTEXT outputFormat: string = "json"
FOR: quarterly_report

// Instead of ambiguous "analyze the sales data from last year"
```

### Model Training
```
COMMAND: train
ON: model
WITH:
  PARAM architecture: string = "transformer"
  PARAM dataset: string = "code_examples.jsonl"
  PARAM epochs: int = 10
  PARAM batchSize: int = 32
  CONTEXT purpose: string = "code completion"
  CONTEXT constraints: list[string] = ["max_memory_gb=16", "max_time_hours=24"]
FOR: production_deployment

// Instead of ambiguous "train a model on my dataset"
```

## Verb Categories

### Primary Verbs
- generate (for creation tasks)
- analyze (for examination tasks)
- transform (for modification tasks)
- validate (for verification tasks)
- train (for learning tasks)
- execute (for runtime tasks)

### Prohibited Constructs
- Relative time references ("yesterday", "next week")
- Ambiguous quantities ("some", "many", "few")
- Implicit subjects ("it", "this", "that")
- Open-ended requests ("if possible", "if you can")
- Unspecified preferences ("better", "good", "nice")

## Type System

### Basic Types
- string: Text values
- int: Integer numbers
- float: Decimal numbers
- bool: True/False values
- daterange: Time period with explicit start and end
- list[type]: Ordered collection of items
- map[keyType,valueType]: Key-value pairs

### Complex Types
- function_signature: {name: string, inputs: list[param], output: type}
- data_schema: {fields: list[field_definition]}
- constraint_set: {rules: list[rule_definition]}

## Error Handling
All errors must be explicitly handled with:
```
ON_ERROR:
  WHEN <condition>: <action>
  DEFAULT: <action>
```

## Context Preservation
```
CONTEXT_SCOPE: <identifier>
CONTEXT_INHERIT: <parent_identifier>
CONTEXT_RESET
```

## Version Control
Every script must start with:
```
AIL_VERSION: 1.0
REQUIRES: <minimum_interpreter_version>
```
