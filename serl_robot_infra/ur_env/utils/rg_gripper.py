import asyncio
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from typing import Union, OrderedDict

class RGGripper:
    def __init__(self, gripper: str, ip: str, port: int = 502) -> None:
        self.client = ModbusClient(ip, port=port, stopbits=1, bytesize=8, parity='E', baudrate=115200, timeout=1)
        if gripper not in ['rg2', 'rg6']:
            raise ValueError("Please specify either 'rg2' or 'rg6'.")
        self.gripper = gripper
        self.max_width = 600 if gripper == 'rg2' else 1600 # original max_width 1100 for rg2
        self.max_force = 400 if gripper == 'rg2' else 1200

    async def connect(self) -> None:
        self.client.connect()

    async def disconnect(self) -> None:
        self.client.close()

    async def activate(self) -> None:
        pass  # No explicit activate command for RG gripper

    async def is_active(self) -> bool:
        status = self.get_status()
        return status[0] == 0  # Not busy

    async def get_current_pressure(self) -> int:
        return self.get_width()

    async def get_object_status(self) -> int:
        status = self.get_status()
        return status[1]  # Grip detected

    async def get_fault_status(self) -> int:
        return 0  # No explicit fault status in RG

    async def automatic_grip(self) -> bool:
        self.close_gripper()

    async def advanced_grip(self, min_pressure: int, max_pressure: int, timeout: int) -> bool:
        self.move_gripper(max_pressure, min_pressure)

    async def continuous_grip(self, timeout: int) -> bool:
        self.close_gripper()

    async def advanced_release(self) -> bool:
        self.open_gripper()

    async def automatic_release(self) -> bool:
        self.open_gripper()

    def get_width(self) -> float:
        result = self.client.read_holding_registers(267, 1, unit=65)
        return result.registers[0] / 10.0

    def get_status(self) -> list:
        result = self.client.read_holding_registers(268, 1, unit=65)
        status = format(result.registers[0], '016b')
        return [int(status[-i-1]) for i in range(7)]

    def close_gripper(self, force_val=400):
        params = [force_val, 0, 16]
        self.client.write_registers(0, params, unit=65)

    def open_gripper(self, force_val=400):
        params = [force_val, self.max_width, 16]
        self.client.write_registers(0, params, unit=65)

    def move_gripper(self, width_val, force_val=400):
        params = [force_val, width_val, 16]
        self.client.write_registers(0, params, unit=65)
