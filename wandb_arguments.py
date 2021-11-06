from dataclasses import dataclass, field
from typing import Optional

@dataclass
class WandBArguments:
    """
    author를 꼭 수정해주세요! 미수정 시 Slave of Booduck이 되버립니다.
    중요 : 위 eval 점수는 단순히 reader의 점수이므로 리더보드의 점수와 다릅니다.
    """
    author: str = field(
        default="nudago",
        metadata={
            "help": "(assential) identifier name"
        },
    )
    name : Optional[str] = field(
        #출력되는 이름 변경
        default=None, #None입력시 id+시간
        metadata={"help": "chart name"},
    )
    tags: Optional[tuple] = field(
        # 원하시는 tags 적용해주세요, 모델이름과 작성자가 자동으로 추가됩니다!
        # 태그 한개추가시 ("xxx",)  <-- ','를 하나 붙혀줘야 오류가 안나옵니다
        default=("experiment","bm25"), 
        metadata={
            "help": "wandB tags list"
        },
    )
    notes: Optional[str] = field(
        # 모델들 돌릴때 잘 까먹을 것 같으면 써보기 
        default=None, 
        metadata={
            "help": "This helps you remember what you were doing when you ran this run."
        },
    ) 
    group: Optional[str] = field(
        default="DPR-N", #default가 None이면 모델이름으로 그룹이 만들어집니다.
        metadata={
            "help": "Group of WandB"
        },
    ) 
    project: str = field(
        #수정 불필요
        default="MRC",
        metadata={
            "help": "Project name of WandB (default : MRC)"
        },
    )
    entity : str = field(
        #수정 불필요
        default="boostcamp-nlp-06",
        metadata={"help" : "Team name of wandB"}
    )

    

