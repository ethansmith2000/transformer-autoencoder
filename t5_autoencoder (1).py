import torch


class Attention(torch.nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.to_qkv = torch.nn.Linear(dim, dim * 3, bias=False)
        self.to_out = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3), (q, k, v))
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(attn_out.shape[0], attn_out.shape[1], -1)
        return self.to_out(attn_out)

class CrossAttention(torch.nn.Module):
    def __init__(self, dim, cross_dim,heads):
        super().__init__()
        self.heads = heads
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_kv = torch.nn.Linear(cross_dim, dim * 2, bias=False)
        self.to_out = torch.nn.Linear(dim, dim)

    def forward(self, x, y):
        q = self.to_q(x)
        k, v = self.to_kv(y).chunk(2, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3), (q, k, v))
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(attn_out.shape[0], attn_out.shape[1], -1)
        return self.to_out(attn_out)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.ff = FeedForward(dim, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

class TransformerBlockWithCross(torch.nn.Module):
    def __init__(self, dim, cross_dim, heads, hidden_dim):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.cross_attn = CrossAttention(dim, cross_dim, heads)
        self.ff = FeedForward(dim, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.norm3 = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.cross_attn(self.norm2(x), y) + x
        x = self.ff(self.norm3(x)) + x
        return x

class AttentionPooler(torch.nn.Module):
    def __init__(self, dim, heads, num_queries):
        super().__init__()
        # there's prob a better init here based on dim, maybe xavier or whatever
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.to_kv = torch.nn.Linear(dim, dim * 2, bias=False)
        self.to_out = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q = self.queries.expand(x.shape[0], -1, -1)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).permute(0, 2, 1, 3), (q, k, v))
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(attn_out.shape[0], attn_out.shape[1], -1)
        return self.to_out(attn_out)

class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim, depth, heads, hidden_dim, num_queries):
        super().__init__()
        self.proj_in = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim),
        )
        self.layers = torch.nn.ModuleList([TransformerBlock(dim, heads, hidden_dim) for _ in range(depth)])
        self.pooler = AttentionPooler(dim, heads, num_queries)
        self.proj_out = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim),
        )

    def forward(self, x):
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pooler(x)
        x = self.proj_out(x)
        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(self, dim, depth, heads, hidden_dim, num_queries, cross_dim):
        super().__init__()
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.layers = torch.nn.ModuleList([TransformerBlockWithCross(dim, cross_dim, heads, hidden_dim) for _ in range(depth)])
        self.proj_out = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim),
        )

    def forward(self, y):
        x = self.queries.expand(y.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, y)
        x = self.proj_out(x)
        return x


class Autoencoder(torch.nn.Module):
    def __init__(self, dim, depth, heads, hidden_dim, num_queries, cross_dim, variational=False, kl_weight=1e-3):
        super().__init__()
        self.encoder = TransformerEncoder(dim, depth, heads, hidden_dim, num_queries)
        self.decoder = TransformerDecoder(dim, depth, heads, hidden_dim, num_queries, cross_dim)
        self.variational = variational
        self.posterior_head = torch.nn.Linear(dim, dim * 2) if variational else None
        self.kl_weight = kl_weight

    def encode(self, x):
        latent = self.encoder(x)
        if not self.variational:
            return latent, None, None

        stats = self.posterior_head(latent)
        mu, logvar = stats.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mu + eps * std 
        return latent, mu, logvar

    def forward(self, x):
        latent, mu, logvar = self.encode(x)
        recon = self.decoder(latent)
        return recon, mu, logvar

    def loss(
        self,
        x,
        recon_loss_fn=None,
    ):
        target = x
        recon_loss_fn = recon_loss_fn or torch.nn.functional.mse_loss

        recon, mu, logvar = self.forward(x)
        recon_loss = recon_loss_fn(recon, target)

        kl_div = None
        if self.variational:
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = recon_loss + self.kl_weight * kl_div
        else:
            total_loss = recon_loss

        return total_loss, {"recon_loss": recon_loss.detach(), "kl_div": None if kl_div is None else kl_div.detach()}
