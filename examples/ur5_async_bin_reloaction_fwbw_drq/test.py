import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import init, unfreeze

# 임베딩 정의
class EmbedDemo(nn.Module):
    def setup(self):
        self.embedding = nn.Embed(num_embeddings=2, features=64)

    def __call__(self, x):
        return self.embedding(x)

# 임베딩 초기화 및 테스트 실행
def main():
    model = EmbedDemo()

    # 예시 입력: 값이 0 또는 1인 (64, 1) shape
    x = jnp.zeros((64, 1), dtype=jnp.int32)  # or jnp.ones((64, 1)) for all 1s
    y = jnp.ones((64, 1), dtype=jnp.int32)  # or jnp.zeros((64, 1)) for all 0s
    # 또는 섞어서
    # x = jnp.array([[i % 2] for i in range(64)], dtype=jnp.int32)

    # 초기화
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, x)
    params = variables['params']
    

    # 임베딩 적용
    outputs_x = model.apply({'params': params}, x)
    outputs_y = model.apply({'params': params}, y)

    print("x 입력 shape:", x.shape)
    print("x 임베딩 출력 shape:", outputs_x.shape)
    print("x[0] 임베딩:", outputs_x[0, 0])

    print("\ny 입력 shape:", y.shape)
    print("y 임베딩 출력 shape:", outputs_y.shape)
    print("y[0] 임베딩:", outputs_y[0, 0])

    # 두 임베딩이 다른지 확인
    diff = jnp.sum((outputs_x[0, 0] - outputs_y[0, 0])**2)
    print("\n임베딩 차이 제곱합 (should be > 0):", diff)

if __name__ == "__main__":
    main()