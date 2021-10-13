from dataclasses import dataclass, field
from typing import Optional

@dataclass
class WandBArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    author: str = field(
        default="SlaveOfBooduck",
        metadata={
            "help": "identifier name"
        },
    )

    project: str = field(
        default="MRC",
        metadata={
            "help": "Project name of WandB (default : MRC)"
        },
    )
    entity : str = field(
        default="boostcamp-nlp-06",
        metadata={"help" : "Team name of wandB"}
    )
    tags: Optional[list] = field(
        default=None, #모델이름은 자동으로 태그로 들어가므로 쓸 필요 없습니다.
        metadata={
            "help": "wandB tags list"
        },
    )
    name : Optional[str] = field(
        default=None, #None입력시 id+시간
        metadata={"help": "chart name"},
    )
    group: Optional[str] = field(
        default=None, #default가 None이면 모델이름으로 그룹이 만들어집니다.
        metadata={
            "help": "Group of WandB"
        },
    ) 
    notes: Optional[str] = field(
        default=None, 
        metadata={
            "help": "A longer description of the run, like a -m commit message in git. This helps you remember what you were doing when you ran this run."
        },
    ) 

    

