"""Step 4: 리스크 관리"""

class RiskManager:
    def __init__(self, stop_loss=-0.01, take_profit=0.003, max_weight=0.1, max_consecutive_loss=3):
        self.stop_loss = stop_loss          # -1%
        self.take_profit = take_profit      # +0.3%
        self.max_weight = max_weight        # 종목당 최대 비중 10%
        self.max_consecutive_loss = max_consecutive_loss  # 연속 손실 허용
        self.consecutive_losses = 0
        self.is_halted = False

    def check_stop_loss(self, current_return: float) -> bool:
        """Stop Loss 체크: True면 청산"""
        return current_return <= self.stop_loss

    def check_take_profit(self, current_return: float) -> bool:
        """목표 수익 달성 체크: True면 청산"""
        return current_return >= self.take_profit

    def should_exit(self, current_return: float) -> tuple:
        """(청산여부, 사유) 반환"""
        if self.is_halted:
            return True, "trading_halted"
        if self.check_stop_loss(current_return):
            return True, "stop_loss"
        if self.check_take_profit(current_return):
            return True, "take_profit"
        return False, "hold"

    def update_loss_count(self, is_loss: bool):
        """연속 손실 카운트 업데이트"""
        if is_loss:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_loss:
                self.is_halted = True
        else:
            self.consecutive_losses = 0
            self.is_halted = False

    def check_position_size(self, weight: float) -> float:
        """종목당 비중 제한"""
        return min(weight, self.max_weight)

    def reset(self):
        self.consecutive_losses = 0
        self.is_halted = False

if __name__ == "__main__":
    print("Step 4: 리스크 관리 모듈 완료")
    rm = RiskManager()
    print(f"  Stop Loss: {rm.stop_loss*100}%")
    print(f"  Take Profit: {rm.take_profit*100}%")
    print(f"  Max Weight: {rm.max_weight*100}%")
    print(f"  Max Consecutive Loss: {rm.max_consecutive_loss}")
