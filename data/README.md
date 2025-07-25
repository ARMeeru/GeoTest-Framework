# Test Data Management

## CSV vs YAML: Design Decision

### Why CSV for Test Data?

We chose **CSV over YAML** for test data management in the GeoTest Framework for several strategic reasons:

#### 1. **Accessibility & Collaboration**
- **Non-technical stakeholders** can easily read, edit, and contribute to test scenarios
- **Business analysts** and **product managers** can review and modify test cases without technical knowledge
- **Excel/Google Sheets compatibility** allows for easy collaboration and bulk editing

#### 2. **Simplicity & Maintainability**
- **Flat structure** eliminates nested hierarchy complexity
- **Column-based organization** makes it easy to scan and understand test scenarios
- **No indentation issues** that can break YAML parsing
- **Bulk operations** are straightforward (sort, filter, find/replace)

#### 3. **Tooling & Processing**
- **Pandas integration** for advanced data manipulation if needed
- **Database-like operations** (filtering, grouping, aggregation)
- **Import/export** from various business tools
- **Data validation** is simpler with tabular structure

#### 4. **Performance & Scale**
- **Faster parsing** than YAML for large datasets
- **Memory efficient** for bulk test data processing
- **Streaming capabilities** for very large test datasets
- **Parallel processing** friendly structure

#### 5. **Testing-Specific Benefits**
- **Parameterized test generation** is straightforward
- **Test result correlation** with input data is natural
- **Reporting integration** - easy to join test results with input scenarios
- **Filtering by test type/priority** using standard CSV operations

### When YAML Might Be Better

YAML would be preferable for:
- **Complex nested configurations** (not applicable to our test scenarios)
- **Multi-line text content** (our test data is simple values)
- **Type safety requirements** (we handle this with Pydantic models)
- **Human-readable config files** (our CSV is plenty readable)

### Our Implementation

```python
# CSV Structure
test_id,test_name,endpoint,input_value,expected_result,test_type,priority,tags,notes

# Pydantic Model Validation
class TestScenario(BaseModel):
    test_id: str
    test_name: str
    endpoint: str
    input_value: str
    expected_result: str
    test_type: str
    priority: str
    tags: Optional[str]
    notes: Optional[str]
```

### Best Practices

#### CSV File Organization
1. **Header row** with clear column names
2. **Consistent data types** within columns
3. **No special characters** in headers (use underscores)
4. **UTF-8 encoding** for international content
5. **Quoted fields** when containing commas or special characters

#### Data Validation
- **Pydantic models** provide type safety and validation
- **Custom validators** ensure data quality
- **Error handling** for malformed CSV data
- **Data cleaning** during import process

#### Version Control
- **Small, focused commits** when updating test data
- **Descriptive commit messages** explaining test scenario changes
- **Review process** for test data modifications
- **Backup strategies** for critical test datasets

### File Structure

```
data/
├── README.md                 # This documentation
├── test_scenarios.csv        # Main test scenario definitions
├── countries_sample.csv      # Sample country data for validation
├── negative_test_cases.csv   # Dedicated negative testing scenarios
└── edge_cases.csv           # Special character and edge case tests
```

### Migration Strategy

If we ever need to migrate to YAML:
1. **Automated conversion** scripts can easily transform CSV to YAML
2. **Pydantic models** remain unchanged (format-agnostic validation)
3. **Test framework** only needs fixture updates, not test logic changes
4. **Gradual migration** is possible (CSV and YAML coexistence)

This design prioritizes **team collaboration**, **maintainability**, and **business stakeholder engagement** over format complexity.