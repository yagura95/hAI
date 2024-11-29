from typing import Dict, List, Any, Optional
from enum import Enum
import ast
import json

class ComponentType(Enum):
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    INFRASTRUCTURE = "infrastructure"

class TechStack:
    def __init__(self):
        self.frameworks = {
            ComponentType.FRONTEND: ["react", "vue", "angular"],
            ComponentType.BACKEND: ["fastapi", "django", "express"],
            ComponentType.DATABASE: ["postgresql", "mongodb", "redis"],
            ComponentType.INFRASTRUCTURE: ["kubernetes", "aws", "gcp"]
        }
        
class CodeGenerator:
    def __init__(self, tech_stack: TechStack):
        self.tech_stack = tech_stack
        self.templates = self.load_templates()
        
    def load_templates(self) -> Dict[str, Any]:
        """Load code templates for different components"""
        return {
            "dashboard": {
                "react": self.load_react_templates(),
                "fastapi": self.load_fastapi_templates(),
                "postgresql": self.load_postgresql_templates()
            }
        }
    
    def generate_project(self, cnl_spec: Dict[str, Any]) -> Dict[str, str]:
        """Generate complete project structure from CNL specification"""
        project = {
            "frontend": self.generate_frontend(cnl_spec),
            "backend": self.generate_backend(cnl_spec),
            "database": self.generate_database(cnl_spec),
            "infrastructure": self.generate_infrastructure(cnl_spec),
            "tests": self.generate_tests(cnl_spec),
            "documentation": self.generate_documentation(cnl_spec)
        }
        return project

class CodeValidator:
    def __init__(self):
        self.security_rules = self.load_security_rules()
        self.performance_rules = self.load_performance_rules()
        
    def validate_generated_code(self, code: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate generated code against security and performance rules"""
        issues = []
        for component, code_str in code.items():
            security_issues = self.check_security(code_str)
            performance_issues = self.check_performance(code_str)
            issues.extend(security_issues + performance_issues)
        return issues

class DeploymentManager:
    def __init__(self, infrastructure: str):
        self.infrastructure = infrastructure
        
    def generate_deployment_files(self, project: Dict[str, str]) -> Dict[str, str]:
        """Generate necessary deployment configuration"""
        return {
            "docker-compose.yml": self.generate_docker_compose(),
            "kubernetes": self.generate_kubernetes_configs(),
            "ci-cd": self.generate_ci_cd_pipeline()
        }

class CNLCompiler:
    def __init__(self):
        self.tech_stack = TechStack()
        self.code_generator = CodeGenerator(self.tech_stack)
        self.validator = CodeValidator()
        self.deployment_manager = DeploymentManager("kubernetes")
        
    def compile(self, cnl_input: str) -> Dict[str, Any]:
        """Main compilation process"""
        try:
            # Parse CNL input
            parsed_spec = self.parse_cnl(cnl_input)
            
            # Generate code
            project = self.code_generator.generate_project(parsed_spec)
            
            # Validate code
            issues = self.validator.validate_generated_code(project)
            if issues:
                self.handle_validation_issues(issues)
            
            # Generate deployment configuration
            deployment = self.deployment_manager.generate_deployment_files(project)
            
            return {
                "project": project,
                "deployment": deployment,
                "validation_issues": issues
            }
            
        except Exception as e:
            raise CNLCompilationError(f"Compilation failed: {str(e)}")
